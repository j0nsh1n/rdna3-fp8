// Throwaway prototype: FP8 E4M3 prefill GEMM for gfx1100 with v_wmma_f32_16x16x16_f16_w32.
// dst[N][M] = scale * sum_k W[M][K] (E4M3 bytes) * Y[N][K] (F32).
// Weights are widened to FP16 (value * 2^-8, same shift as the decode prototype) while staging into LDS, so the
// widening is paid once per BN columns. Activations are converted to FP16 once per call in the same pair order:
// within each group of 4 K values the order is k0, k2, k1, k3 for both operands, which leaves the dot product unchanged.
//
// Prototype limits: M % 128 == 0, K % 32 == 0; N is padded up to a multiple of BN on the host.
// The WMMA output layout used in the store (element i of lane t -> row 2i + t/16 of A, column t%16 of B) is from
// AMD's RDNA 3 WMMA guide, not yet checked on this card; the CPU comparison below checks it.
//
// NaN codes (0x7F, 0xFF) make the whole output row NaN, as in llama.cpp: a per-word check while staging.
//
// Usage: fp8_gemm <N> [M K reps plain|nan v1|v2]; "nan" plants 0x7F in row 5 and 0xFF in the last row, which must be NaN in
// every column while no other output is.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int THREADS = 256;               // 8 waves
constexpr int WM = 64, WN = 32;            // per-wave tile: 2 x 4 waves over the block
constexpr int TM = WM / 16, TN = WN / 16;  // WMMA tiles per wave
constexpr int LDS_STRIDE = BK + 8;         // halves per LDS row; padding against bank conflicts

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8), aligned(16)));

// Whole-register reinterpretation; memcpy into half16_t made LLVM move each 16-bit half separately.
__device__ __forceinline__ half16_t lds_frag(const uint32_t * p) {
    return __builtin_bit_cast(half16_t, *reinterpret_cast<const u32x8 *>(p));
}

__device__ __forceinline__ uint32_t widen_even(uint32_t x) {
    return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3F803F80u);
}

__device__ __forceinline__ uint32_t widen_odd(uint32_t x) {
    return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u);
}

// Non-zero exactly when some byte is a NaN code (checked over all 2^32 words by proto/widen_check.cpp).
__device__ __forceinline__ uint32_t nan_bytes(uint32_t x) {
    return ((x & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
}

__global__ void act_to_f16_pairs(const float * __restrict__ y, uint32_t * __restrict__ yh, size_t words) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= words) {
        return;
    }
    const float4 a = reinterpret_cast<const float4 *>(y)[i];
    const __half2 ev = __floats2half2_rn(a.x, a.z);
    const __half2 od = __floats2half2_rn(a.y, a.w);
    uint32_t e, o;
    std::memcpy(&e, &ev, 4);
    std::memcpy(&o, &od, 4);
    yh[2 * i] = e;
    yh[2 * i + 1] = o;
}

__global__ __launch_bounds__(THREADS) void gemm_fp8(const uint8_t * __restrict__ w, const uint32_t * __restrict__ yh,
                                                   float * __restrict__ dst, int M, int N, int K, float out_scale) {
    __shared__ uint32_t sa[BM * LDS_STRIDE / 2];
    __shared__ uint32_t sb[BN * LDS_STRIDE / 2];
    __shared__ uint32_t row_nan[BM];

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32;
    const int wm0 = (wave / 4) * WM;  // 2 waves along M
    const int wn0 = (wave % 4) * WN;  // 4 waves along N

    float8_t acc[TM][TN] = {};
    uint32_t nanf = 0;  // this thread always stages the same weight row, so one flag covers its share of that row

    // Staging: A is 128 rows x 32 bytes = 256 x 16 bytes, one chunk per thread.
    //          B is 128 rows x 32 halves = 256 x 32 bytes, two 16-byte chunks per thread.
    const int a_row = tid / 2, a_half = tid % 2;
    const int b_row = tid / 2, b_half = tid % 2;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const u32x4 wv = __builtin_nontemporal_load(
            reinterpret_cast<const u32x4 *>(w + (size_t) (m0 + a_row) * K + k0 + a_half * 16));
        const uint4 * yp = reinterpret_cast<const uint4 *>(yh + ((size_t) (n0 + b_row) * K + k0 + b_half * 16) / 2);
        const uint4 y0 = yp[0], y1 = yp[1];

        uint32_t * ad = sa + (a_row * LDS_STRIDE + a_half * 16) / 2;
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
        nanf |= nan_bytes(ws[0]) | nan_bytes(ws[1]) | nan_bytes(ws[2]) | nan_bytes(ws[3]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            ad[2 * q] = widen_even(ws[q]);
            ad[2 * q + 1] = widen_odd(ws[q]);
        }
        uint4 * bd = reinterpret_cast<uint4 *>(sb + (b_row * LDS_STRIDE + b_half * 16) / 2);
        bd[0] = y0;
        bd[1] = y1;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16_t af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                af[i] = lds_frag(sa + ((wm0 + i * 16 + lane % 16) * LDS_STRIDE + kk) / 2);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bf[j] = lds_frag(sb + ((wn0 + j * 16 + lane % 16) * LDS_STRIDE + kk) / 2);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // The two threads staging a row are adjacent lanes of one wave (a_half = tid % 2).
    nanf |= __shfl_xor(nanf, 1, 32);
    if (a_half == 0) {
        row_nan[a_row] = nanf;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + wn0 + j * 16 + lane % 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int ml = wm0 + i * 16 + 2 * e + lane / 16;
                const int m = m0 + ml;
                if (n < N) {
                    dst[(size_t) n * M + m] = row_nan[ml] ? NAN : acc[i][j][e] * out_scale;
                }
            }
        }
    }
}

// Version 2: the next K tile is fetched from global memory into registers while the current one is computed, and LDS
// is double-buffered, so there is one barrier per K step and global latency overlaps the WMMA work.
__global__ __launch_bounds__(THREADS) void gemm_fp8_v2(const uint8_t * __restrict__ w, const uint32_t * __restrict__ yh,
                                                      float * __restrict__ dst, int M, int N, int K, float out_scale) {
    __shared__ uint32_t sa[2][BM * LDS_STRIDE / 2];
    __shared__ uint32_t sb[2][BN * LDS_STRIDE / 2];
    __shared__ uint32_t row_nan[BM];

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32;
    const int wm0 = (wave / 4) * WM;
    const int wn0 = (wave % 4) * WN;
    const int a_row = tid / 2, a_half = tid % 2;
    const int b_row = tid / 2, b_half = tid % 2;

    const uint8_t * wp = w + (size_t) (m0 + a_row) * K + a_half * 16;
    const uint32_t * yq = yh + ((size_t) (n0 + b_row) * K + b_half * 16) / 2;

    float8_t acc[TM][TN] = {};
    uint32_t nanf = 0;

    u32x4 wv = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(wp));
    uint4 y0 = reinterpret_cast<const uint4 *>(yq)[0], y1 = reinterpret_cast<const uint4 *>(yq)[1];

    auto stage = [&](int buf) {
        uint32_t * ad = sa[buf] + (a_row * LDS_STRIDE + a_half * 16) / 2;
        const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
        nanf |= nan_bytes(ws[0]) | nan_bytes(ws[1]) | nan_bytes(ws[2]) | nan_bytes(ws[3]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            ad[2 * q] = widen_even(ws[q]);
            ad[2 * q + 1] = widen_odd(ws[q]);
        }
        uint4 * bd = reinterpret_cast<uint4 *>(sb[buf] + (b_row * LDS_STRIDE + b_half * 16) / 2);
        bd[0] = y0;
        bd[1] = y1;
    };

    stage(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            wv = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(wp + k0 + BK));
            const uint4 * yp = reinterpret_cast<const uint4 *>(yq + (k0 + BK) / 2);
            y0 = yp[0];
            y1 = yp[1];
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16_t af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                af[i] = lds_frag(sa[buf] + ((wm0 + i * 16 + lane % 16) * LDS_STRIDE + kk) / 2);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bf[j] = lds_frag(sb[buf] + ((wn0 + j * 16 + lane % 16) * LDS_STRIDE + kk) / 2);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], acc[i][j]);
                }
            }
        }
        if (more) {
            stage(buf ^ 1);
        }
        __syncthreads();
        buf ^= 1;
    }

    nanf |= __shfl_xor(nanf, 1, 32);
    if (a_half == 0) {
        row_nan[a_row] = nanf;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + wn0 + j * 16 + lane % 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int ml = wm0 + i * 16 + 2 * e + lane / 16;
                const int m = m0 + ml;
                if (n < N) {
                    dst[(size_t) n * M + m] = row_nan[ml] ? NAN : acc[i][j][e] * out_scale;
                }
            }
        }
    }
}


__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    uint32_t r;
    std::memcpy(&r, &h, 4);
    return r;
}

// Generic version: template tile sizes, operands swapped in the WMMA (activations first) so the accumulator lanes run
// along M and each store instruction writes two 64-byte runs of dst instead of 32 scattered words, optional split-K
// (each z slice writes its own F32 partial plane, summed by splitk_reduce), NBUF = 1 (one LDS buffer, two barriers per
// K step) or 2 (double-buffered). The NaN flag of a weight row is OR-ed through LDS atomics, so any thread/row mapping
// works.
//   BM, BN, BK: block tile; WAM x WAN: waves along M and N; PAD: extra halves per LDS row (multiple of 8).
template <int BM, int BN, int BK, int WAM, int WAN, int PAD, int NBUF, bool VSTORE = true, bool PEEL = true, int DIAG = 0, bool NT = true, bool FUSE = false, bool FIXUP = false, int RAST = 0>
__global__ __launch_bounds__(WAM * WAN * 32) void gemm_t(const uint8_t * __restrict__ w, const uint32_t * __restrict__ yh,
                                                         float * __restrict__ dst, float * __restrict__ ws, int * __restrict__ cnt, int M, int N, int K, int kper,
                                                         size_t zstride, float out_scale) {
    constexpr int T = WAM * WAN * 32;
    constexpr int WM = BM / WAM, WN = BN / WAN, TM = WM / 16, TN = WN / 16;
    constexpr int S = BK + PAD;  // halves per LDS row
    constexpr int CA = BK / 16, CB = BK / 8;
    constexpr int SA = BM * CA / T, SB = BN * CB / T;
    static_assert(BM * CA % T == 0 && BN * CB % T == 0 && SA >= 1 && SB >= 1, "staging does not divide");
    static_assert(PAD % 8 == 0 && BK % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "bad shape");
    __shared__ uint32_t sa[NBUF][BM * S / 2];
    __shared__ uint32_t sb[NBUF][BN * S / 2];
    __shared__ uint32_t row_nan[BM];

    // RAST 0: grid (M tiles, N tiles), neighbouring blocks share the activation tile.
    // RAST 1: 1-D grid, N tiles fastest, so the blocks that run together share one weight tile.
    int tile_m = blockIdx.x, tile_n = blockIdx.y;
    if constexpr (RAST == 1) {
        const int tiles_n = (N + BN - 1) / BN;
        tile_m = blockIdx.x / tiles_n;
        tile_n = blockIdx.x % tiles_n;
    }
    const int m0 = tile_m * BM;
    const int n0 = tile_n * BN;
    const int kbeg = blockIdx.z * kper;
    const int tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32;
    const int wm0 = (wave % WAM) * WM;
    const int wn0 = (wave / WAM) * WN;

    // One base pointer per operand; slot i is a uniform number of rows further on, so the compiler needs no per-slot
    // pointer registers.
    constexpr int RA = T / CA, RB = T / CB;
    const int arow0 = tid / CA;
    const uint8_t * wbase = w + (size_t) (m0 + arow0) * K + (tid % CA) * 16 + kbeg;
    const size_t wstep = (size_t) RA * K;
    uint32_t * adst0 = &sa[0][0] + (arow0 * S + (tid % CA) * 16) / 2;
    // FUSE: yh really holds the F32 activations and they are converted to FP16 pairs while staging.
    const uint32_t * ybase = yh + ((size_t) (n0 + tid / CB) * K + (tid % CB) * 8 + kbeg) / (FUSE ? 1 : 2);
    const size_t ystep = (size_t) RB * K / (FUSE ? 1 : 2);
    uint32_t * bdst0 = &sb[0][0] + ((tid / CB) * S + (tid % CB) * 8) / 2;
    for (int i = tid; i < BM; i += T) {
        row_nan[i] = 0;
    }

    float8_t acc[TN][TM] = {};
    uint32_t nanf[SA] = {};
    u32x4 wv[SA];
    u32x4 yv[SB * (FUSE ? 2 : 1)];

    auto load = [&](int off) {
        if constexpr (DIAG == 1) {
            return;  // diagnostic: no global traffic after the first tile; results are wrong
        }
#pragma unroll
        for (int i = 0; i < SA; ++i) {
            const u32x4 * wsrc = reinterpret_cast<const u32x4 *>(wbase + i * wstep + off);
            if constexpr (NT) {
                wv[i] = __builtin_nontemporal_load(wsrc);
            } else {
                wv[i] = *wsrc;
            }
        }
#pragma unroll
        for (int i = 0; i < SB; ++i) {
            if constexpr (FUSE) {
                const u32x4 * src = reinterpret_cast<const u32x4 *>(ybase + i * ystep + off);
                yv[2 * i] = src[0];
                yv[2 * i + 1] = src[1];
            } else {
                yv[i] = *reinterpret_cast<const u32x4 *>(ybase + i * ystep + off / 2);
            }
        }
    };
    auto stage = [&](int buf) {
        const int boff = buf * (BM * S / 2);
        const int boffb = buf * (BN * S / 2);
#pragma unroll
        for (int i = 0; i < SA; ++i) {
            u32x4 * ad = reinterpret_cast<u32x4 *>(adst0 + i * (RA * S / 2) + boff);
            const uint32_t ws[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
            nanf[i] |= nan_bytes(ws[0]) | nan_bytes(ws[1]) | nan_bytes(ws[2]) | nan_bytes(ws[3]);
            if constexpr (DIAG == 2) {
                ad[0] = u32x4{ws[0], ws[1], ws[2], ws[3]};  // diagnostic: no widening; results are wrong
                ad[1] = u32x4{ws[3], ws[2], ws[1], ws[0]};
            } else if constexpr (VSTORE) {
                ad[0] = u32x4{widen_even(ws[0]), widen_odd(ws[0]), widen_even(ws[1]), widen_odd(ws[1])};
                ad[1] = u32x4{widen_even(ws[2]), widen_odd(ws[2]), widen_even(ws[3]), widen_odd(ws[3])};
            } else {
                uint32_t * ad32 = reinterpret_cast<uint32_t *>(ad);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    ad32[2 * q] = widen_even(ws[q]);
                    ad32[2 * q + 1] = widen_odd(ws[q]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < SB; ++i) {
            if constexpr (FUSE) {
                const float4 g0 = __builtin_bit_cast(float4, yv[2 * i]), g1 = __builtin_bit_cast(float4, yv[2 * i + 1]);
                *reinterpret_cast<u32x4 *>(bdst0 + i * (RB * S / 2) + boffb) =
                    u32x4{pack2(g0.x, g0.z), pack2(g0.y, g0.w), pack2(g1.x, g1.z), pack2(g1.y, g1.w)};
            } else {
                *reinterpret_cast<u32x4 *>(bdst0 + i * (RB * S / 2) + boffb) = yv[i];
            }
        }
    };
    auto compute = [&](int buf) {
        if constexpr (DIAG == 3) {
            return;  // diagnostic: no LDS reads or WMMA; results are wrong
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16_t wf[TM], yf[TN];
#pragma unroll
            for (int b = 0; b < TM; ++b) {
                wf[b] = lds_frag(sa[buf] + ((wm0 + b * 16 + lane % 16) * S + kk) / 2);
            }
#pragma unroll
            for (int a = 0; a < TN; ++a) {
                yf[a] = lds_frag(sb[buf] + ((wn0 + a * 16 + lane % 16) * S + kk) / 2);
            }
#pragma unroll
            for (int a = 0; a < TN; ++a) {
#pragma unroll
                for (int b = 0; b < TM; ++b) {
                    acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(yf[a], wf[b], acc[a][b]);
                }
            }
        }
    };

    load(0);
    if constexpr (NBUF == 2) {
        stage(0);
        __syncthreads();
        int buf = 0;
        if constexpr (PEEL) {
            for (int k = 0; k + BK < kper; k += BK) {
                load(k + BK);
                compute(buf);
                stage(buf ^ 1);
                __syncthreads();
                buf ^= 1;
            }
            compute(buf);
        } else {
            for (int k = 0; k < kper; k += BK) {
                const bool more = k + BK < kper;
                if (more) {
                    load(k + BK);
                }
                compute(buf);
                if (more) {
                    stage(buf ^ 1);
                }
                __syncthreads();
                buf ^= 1;
            }
        }
    } else {
        if constexpr (PEEL) {
            for (int k = 0; k + BK < kper; k += BK) {
                stage(0);
                __syncthreads();
                load(k + BK);
                compute(0);
                __syncthreads();
            }
            stage(0);
            __syncthreads();
            compute(0);
            __syncthreads();
        } else {
            for (int k = 0; k < kper; k += BK) {
                stage(0);
                __syncthreads();
                if (k + BK < kper) {
                    load(k + BK);
                }
                compute(0);
                __syncthreads();
            }
        }
    }

#pragma unroll
    for (int i = 0; i < SA; ++i) {
        atomicOr(&row_nan[arow0 + i * RA], nanf[i]);
    }
    __syncthreads();

    // Split-K: without FIXUP every slice writes its own plane of dst (summed by splitk_reduce). With FIXUP the slices
    // write partials to ws, and the last slice of a tile to finish adds the others to its registers and writes dst, so
    // no second launch is needed. cnt holds one counter per tile, zero between launches (the last slice resets it).
    float * out = (FIXUP ? ws : dst) + blockIdx.z * zstride;
#pragma unroll
    for (int b = 0; b < TM; ++b) {
        const int ml = wm0 + b * 16 + lane % 16;
        const bool isnan_row = row_nan[ml] != 0;
        const int m = m0 + ml;
#pragma unroll
        for (int a = 0; a < TN; ++a) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[a][b][e] = isnan_row ? NAN : acc[a][b][e] * out_scale;
                const int n = n0 + wn0 + a * 16 + 2 * e + lane / 16;
                if (n < N) {
                    out[(size_t) n * M + m] = acc[a][b][e];
                }
            }
        }
    }
    if constexpr (FIXUP) {
        __shared__ int last;
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int tile = tile_n * (M / BM) + tile_m;
            const int old = atomicAdd(&cnt[tile], 1);
            last = old == (int) gridDim.z - 1;
            if (last) {
                cnt[tile] = 0;
            }
        }
        __syncthreads();
        if (last) {
            __threadfence();
#pragma unroll
            for (int b = 0; b < TM; ++b) {
                const int m = m0 + wm0 + b * 16 + lane % 16;
#pragma unroll
                for (int a = 0; a < TN; ++a) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int n = n0 + wn0 + a * 16 + 2 * e + lane / 16;
                        if (n < N) {
                            float sum = acc[a][b][e];
                            for (int z = 0; z < (int) gridDim.z; ++z) {
                                if (z != (int) blockIdx.z) {
                                    sum += __builtin_nontemporal_load(ws + z * zstride + (size_t) n * M + m);
                                }
                            }
                            dst[(size_t) n * M + m] = sum;
                        }
                    }
                }
            }
        }
    }
}

__global__ void splitk_reduce(const float4 * __restrict__ ws, float4 * __restrict__ out, size_t n4, int splits) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    float4 s = ws[i];
    for (int z = 1; z < splits; ++z) {
        const float4 v = ws[i + z * n4];
        s.x += v.x;
        s.y += v.y;
        s.z += v.z;
        s.w += v.w;
    }
    out[i] = s;
}

struct Ctx {
    const uint8_t * w;
    const uint32_t * yh;
    const float * yf;
    float * dd;
    float * ws;  // split-K planes
    int * cnt;   // split-K tile counters (FIXUP), zero between launches
    int M, N, K;
    float out_scale;
};

template <int BM, int BN, int BK, int WAM, int WAN, int PAD, int NBUF, bool VSTORE = true, bool PEEL = true, int DIAG = 0, bool NT = true, bool FUSE = false, bool FIXUP = false, int RAST = 0>
static void launch_t(const Ctx & c, int splits) {
    const int gy = (c.N + BN - 1) / BN;
    const int kper = c.K / splits;
    const uint32_t * y = FUSE ? reinterpret_cast<const uint32_t *>(c.yf) : c.yh;
    if (splits == 1) {
        gemm_t<BM, BN, BK, WAM, WAN, PAD, NBUF, VSTORE, PEEL, DIAG, NT, FUSE, FIXUP, RAST><<<(RAST == 1 ? dim3(c.M / BM * gy, 1, 1) : dim3(c.M / BM, gy, 1)), WAM * WAN * 32>>>(c.w, y, c.dd, nullptr, nullptr, c.M, c.N, c.K, kper, 0, c.out_scale);
    } else if (FIXUP) {
        const size_t plane = (size_t) c.N * c.M;
        gemm_t<BM, BN, BK, WAM, WAN, PAD, NBUF, VSTORE, PEEL, DIAG, NT, FUSE, FIXUP, RAST><<<(RAST == 1 ? dim3(c.M / BM * gy, 1, splits) : dim3(c.M / BM, gy, splits)), WAM * WAN * 32>>>(c.w, y, c.dd, c.ws, c.cnt, c.M, c.N, c.K, kper, plane, c.out_scale);
    } else {
        const size_t plane = (size_t) c.N * c.M;
        gemm_t<BM, BN, BK, WAM, WAN, PAD, NBUF, VSTORE, PEEL, DIAG, NT, FUSE, FIXUP, RAST><<<(RAST == 1 ? dim3(c.M / BM * gy, 1, splits) : dim3(c.M / BM, gy, splits)), WAM * WAN * 32>>>(c.w, y, c.ws, nullptr, nullptr, c.M, c.N, c.K, kper, plane, c.out_scale);
        const size_t n4 = plane / 4;
        splitk_reduce<<<(n4 + 255) / 256, 256>>>(reinterpret_cast<const float4 *>(c.ws), reinterpret_cast<float4 *>(c.dd), n4, splits);
    }
}

static void launch_v1(const Ctx & c, int) {
    gemm_fp8<<<dim3(c.M / BM, (c.N + BN - 1) / BN), THREADS>>>(c.w, c.yh, c.dd, c.M, c.N, c.K, c.out_scale);
}
static void launch_v2(const Ctx & c, int) {
    gemm_fp8_v2<<<dim3(c.M / BM, (c.N + BN - 1) / BN), THREADS>>>(c.w, c.yh, c.dd, c.M, c.N, c.K, c.out_scale);
}

struct Var {
    const char * name;
    int bm, bn, splits;
    void (*fn)(const Ctx &, int);
    bool fused = false;  // converts the F32 activations itself, so the separate conversion pass is skipped
};

// Name: t<BM>x<BN>x<BK>_w<WAM>x<WAN>_p<PAD>_b<NBUF>[_s<splits>]
#define VT(BM_, BN_, BK_, WAM_, WAN_, PAD_, NB_, SP_, NAME_) \
    {NAME_, BM_, BN_, SP_, launch_t<BM_, BN_, BK_, WAM_, WAN_, PAD_, NB_, true, false, 0, false>}
static const Var VARS[] = {
    {"v1", 128, 128, 1, launch_v1},
    {"v2", 128, 128, 1, launch_v2},
    VT(128, 128, 32, 2, 4, 8, 2, 1, "t128x128x32"),
    VT(128, 128, 64, 2, 4, 8, 1, 1, "t128x128x64"),
    VT(128, 128, 32, 2, 4, 0, 2, 1, "t128x128x32_p0"),
    VT(128, 128, 32, 2, 4, 8, 1, 1, "t128x128x32_b1"),
    VT(64, 64, 32, 2, 2, 8, 2, 1, "t64x64x32"),
    VT(64, 64, 64, 2, 2, 8, 2, 1, "t64x64x64"),
    VT(128, 64, 32, 2, 2, 8, 2, 1, "t128x64x32"),
    VT(64, 128, 32, 2, 2, 8, 2, 1, "t64x128x32"),
    VT(128, 128, 32, 4, 2, 8, 2, 1, "t128x128x32_w4x2"),
    VT(128, 128, 32, 2, 2, 8, 2, 1, "t128x128x32_4w"),
    VT(128, 256, 32, 2, 4, 8, 2, 1, "t128x256x32"),
    VT(256, 128, 32, 4, 2, 8, 2, 1, "t256x128x32"),
    VT(64, 64, 32, 2, 2, 8, 2, 2, "t64x64x32_s2"),
    VT(64, 64, 32, 2, 2, 8, 2, 4, "t64x64x32_s4"),
    VT(128, 128, 32, 2, 4, 8, 2, 2, "t128x128x32_s2"),
    VT(128, 128, 32, 2, 4, 8, 2, 4, "t128x128x32_s4"),
    VT(128, 64, 32, 2, 2, 8, 2, 2, "t128x64x32_s2"),
    VT(128, 64, 32, 2, 2, 8, 2, 4, "t128x64x32_s4"),
    VT(128, 128, 64, 4, 2, 8, 1, 1, "t128x128x64_w4x2"),
    // Switch comparisons on t128x128x64_w4x2: NT = non-temporal weight loads, peel = last K step peeled out of the
    // loop, novs = scalar LDS stores of the widened weights; dg1/2/3 are timing diagnostics with wrong results
    // (no global loads in the loop / no widening / no LDS reads and WMMA).
    {"t128x128x64_w4x2_nt", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 0, true>},
    {"t128x128x64_w4x2_peel", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, true, 0, false>},
    {"t128x128x64_w4x2_novs", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, false, false, 0, false>},
    {"f128x128x64_w4x2", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 0, false, true>, true},
    {"f64x128x64", 64, 128, 1, launch_t<64, 128, 64, 2, 2, 8, 1, true, false, 0, false, true>, true},
    {"f128x128x64_s2", 128, 128, 2, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, true>, true},
    {"f128x128x64_s4", 128, 128, 4, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, true>, true},
    {"f128x256x64", 128, 256, 1, launch_t<128, 256, 64, 2, 4, 8, 1, true, false, 0, false, true>, true},
    {"x128x128x64_s2", 128, 128, 2, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, false, true>},
    {"x128x128x64_s3", 128, 128, 3, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, false, true>},
    {"x128x128x64_s4", 128, 128, 4, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, false, true>},
    {"x128x128x64_s6", 128, 128, 6, launch_t<128, 128, 64, 2, 4, 8, 1, true, false, 0, false, false, true>},
    {"x64x128x64_s2", 64, 128, 2, launch_t<64, 128, 64, 2, 2, 8, 1, true, false, 0, false, false, true>},
    {"x64x128x64_s4", 64, 128, 4, launch_t<64, 128, 64, 2, 2, 8, 1, true, false, 0, false, false, true>},
    {"r128x128x64_w4x2", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 0, false, false, false, 1>},
    {"r128x256x64", 128, 256, 1, launch_t<128, 256, 64, 2, 4, 8, 1, true, false, 0, false, false, false, 1>},
    {"r64x128x64", 64, 128, 1, launch_t<64, 128, 64, 2, 2, 8, 1, true, false, 0, false, false, false, 1>},
    {"r128x128x64_p16", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 16, 1, true, false, 0, false, false, false, 0>},
    {"r128x128x64_p0", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 0, 1, true, false, 0, false, false, false, 0>},
    {"dg1_noload", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 1, false>},
    {"dg2_nowiden", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 2, false>},
    {"dg3_nocompute", 128, 128, 1, launch_t<128, 128, 64, 4, 2, 8, 1, true, false, 3, false>},
    VT(64, 128, 64, 2, 2, 8, 1, 1, "t64x128x64"),
    VT(128, 64, 64, 2, 2, 8, 1, 1, "t128x64x64"),
    VT(64, 64, 64, 2, 2, 8, 1, 1, "t64x64x64_b1"),
    VT(256, 128, 64, 4, 2, 8, 1, 1, "t256x128x64"),
    VT(128, 256, 64, 2, 4, 8, 1, 1, "t128x256x64"),
    VT(128, 64, 64, 2, 2, 8, 2, 1, "t128x64x64_b2"),
    VT(64, 128, 64, 2, 2, 8, 2, 1, "t64x128x64_b2"),
    VT(64, 64, 128, 2, 2, 8, 1, 1, "t64x64x128"),
    VT(128, 64, 128, 2, 2, 8, 1, 1, "t128x64x128"),
    VT(64, 128, 128, 2, 2, 8, 1, 1, "t64x128x128"),
    VT(128, 64, 128, 4, 2, 8, 1, 1, "t128x64x128_w4x2"),
    VT(128, 128, 64, 2, 4, 8, 1, 2, "t128x128x64_s2"),
    VT(128, 128, 64, 2, 4, 8, 1, 4, "t128x128x64_s4"),
    VT(64, 64, 64, 2, 2, 8, 2, 2, "t64x64x64_s2"),
    VT(64, 64, 64, 2, 2, 8, 2, 4, "t64x64x64_s4"),
};
static const int NVARS = sizeof(VARS) / sizeof(VARS[0]);

static float e4m3_to_float(uint8_t x) {
    const int ax = x & 0x7F;
    const int e = ax >> 3, m = ax & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return x & 0x80 ? -v : v;
}

static void require_gfx1100() {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1100", 7) != 0) {
        std::fprintf(stderr, "device 0 is %s (%s), not the gfx1100 card; refusing to time the wrong GPU\n", prop.name, prop.gcnArchName);
        std::exit(3);
    }
}

static const Var * find_var(const char * name) {
    for (int i = 0; i < NVARS; ++i) {
        if (std::strcmp(VARS[i].name, name) == 0) {
            return &VARS[i];
        }
    }
    return nullptr;
}

// Median of per-launch event pairs (the original method) and the mean over a batch of back-to-back launches.
struct Timing {
    double med_us, min_us, p90_us, batch_us;
};
template <class F>
static Timing time_fn(F && f, int reps) {
    double burst_us = 0;
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    // Ramp the clocks first: a 100 ms burst of back-to-back launches, whose mean also sizes the batches below.
    int burst = 0;
    {
        CHECK(hipEventRecord(t0));
        float m = 0;
        do {
            for (int i = 0; i < 10; ++i) {
                f();
            }
            burst += 10;
            CHECK(hipEventRecord(t1));
            CHECK(hipEventSynchronize(t1));
            CHECK(hipEventElapsedTime(&m, t0, t1));
        } while (m < 100.0f);
        burst_us = m * 1e3 / burst;
    }
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
        CHECK(hipEventRecord(t0));
        f();
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float m;
        CHECK(hipEventElapsedTime(&m, t0, t1));
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.med_us = ms[ms.size() / 2] * 1e3;
    t.min_us = ms.front() * 1e3;
    t.p90_us = ms[ms.size() * 9 / 10] * 1e3;
    // Median over 9 batches of back-to-back launches: no CPU launch gap in the interval, robust to one slow batch.
    const int B = std::max(5, (int) (4000.0 / burst_us));  // each batch lasts about 4 ms
    std::vector<double> bm;
    for (int r = 0; r < 9; ++r) {
        CHECK(hipEventRecord(t0));
        for (int i = 0; i < B; ++i) {
            f();
        }
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float m;
        CHECK(hipEventElapsedTime(&m, t0, t1));
        bm.push_back(m * 1e3 / B);
    }
    std::sort(bm.begin(), bm.end());
    t.batch_us = bm[bm.size() / 2];
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    return t;
}

static double tflops(int M, int N, int K, double us) {
    return 2.0 * M * N * (double) K / (us * 1e6);
}

int main(int argc, char ** argv) {
    require_gfx1100();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <N> [M K reps plain|nan|bench variant...]\n", argv[0]);
        return 2;
    }
    const int N = std::atoi(argv[1]);
    const int M = argc > 2 ? std::atoi(argv[2]) : 16384;
    const int K = argc > 3 ? std::atoi(argv[3]) : 14336;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 50;
    const bool nan_mode = argc > 5 && std::strcmp(argv[5], "nan") == 0;
    const bool bench = argc > 5 && std::strcmp(argv[5], "bench") == 0;
    std::vector<const Var *> sel;
    for (int a = 6; a < argc; ++a) {
        if (std::strcmp(argv[a], "all") == 0) {
            for (int i = 0; i < NVARS; ++i) {
                sel.push_back(&VARS[i]);
            }
        } else if (const Var * v = find_var(argv[a])) {
            sel.push_back(v);
        } else {
            std::fprintf(stderr, "unknown variant %s\n", argv[a]);
            return 2;
        }
    }
    if (sel.empty() && !bench) {
        sel.push_back(find_var("v1"));
    }
    if (M % 256 != 0 || K % 128 != 0 || N < 1) {
        std::fprintf(stderr, "need M %% 256 == 0, K %% 128 == 0, N >= 1\n");
        return 2;
    }
    const int Np = (N + 255) / 256 * 256;
    const float tensor_scale = 0.0123f;
    const float out_scale = tensor_scale * 256.0f;

    std::mt19937 rng(1234);
    std::vector<uint8_t> hw((size_t) M * K);
    for (auto & b : hw) {
        do {
            b = (uint8_t) rng();
        } while ((b & 0x7F) == 0x7F);
    }
    if (nan_mode) {
        hw[(size_t) 5 * K + 77] = 0x7F;
        hw[(size_t) (M - 1) * K + K - 1] = 0xFF;
    }
    std::vector<float> hy((size_t) Np * K, 0.0f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t) N * K; ++i) {
        hy[i] = nd(rng);
    }

    uint8_t * dw;
    float *dy, *dd, *dws;
    uint32_t * dyh;
    CHECK(hipMalloc(&dw, hw.size()));
    CHECK(hipMalloc(&dy, hy.size() * 4));
    CHECK(hipMalloc(&dyh, hy.size() * 2));
    CHECK(hipMalloc(&dd, (size_t) Np * M * 4));
    CHECK(hipMalloc(&dws, (size_t) Np * M * 4 * 8));
    int * dcnt;
    CHECK(hipMalloc(&dcnt, 1 << 20));
    CHECK(hipMemset(dcnt, 0, 1 << 20));
    CHECK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dy, hy.data(), hy.size() * 4, hipMemcpyHostToDevice));

    const size_t words = (size_t) Np * K / 4;
    // N is passed as the real N so the kernel's store guard works; the grid covers ceil(N / BN) tiles.
    const Ctx ctx{dw, dyh, dy, dd, dws, dcnt, M, N, K, out_scale};
    auto conv = [&] { act_to_f16_pairs<<<(words + 255) / 256, 256>>>(dy, dyh, words); };

    float lut[256];
    for (int c = 0; c < 256; ++c) {
        lut[c] = e4m3_to_float((uint8_t) c);
    }
    std::vector<float> got((size_t) N * M);
    int rc = 0;
    auto check = [&](const char * name) {
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(got.data(), dd, got.size() * 4, hipMemcpyDeviceToHost));
        long nan_missing = 0, nan_unexpected = 0;
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                const bool want = nan_mode && (m == 5 || m == M - 1);
                const bool is = std::isnan(got[(size_t) n * M + m]);
                nan_missing += want && !is;
                nan_unexpected += !want && is;
            }
        }
        std::printf("[%s] NaN outputs: missing %ld, unexpected %ld (%s)\n", name, nan_missing, nan_unexpected,
                    nan_mode ? "rows 5 and M-1 planted" : "none planted");
        double err2 = 0, ref2 = 0, max_rel = 0;
        std::mt19937 srng(99);
        std::uniform_int_distribution<int> pm(0, M - 1), pn(0, N - 1);
        const int samples = 4096;
        for (int s = 0; s < samples; ++s) {
            const int m = s < 256 ? s % M : pm(srng);
            const int n = s < 256 ? (s / 16) % N : pn(srng);
            if (nan_mode && (m == 5 || m == M - 1)) {
                continue;
            }
            double ref = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                const double p = (double) lut[hw[(size_t) m * K + k]] * (double) (_Float16) hy[(size_t) n * K + k];
                ref += p;
                mag += std::fabs(p);
            }
            ref *= tensor_scale;
            mag *= tensor_scale;
            const double d = got[(size_t) n * M + m] - ref;
            err2 += d * d;
            ref2 += ref * ref;
            max_rel = std::max(max_rel, std::fabs(d) / mag);
        }
        const double nmse = err2 / ref2;
        std::printf("[%s] N=%d M=%d K=%d: sampled NMSE %.3e, max |err|/sum|w*a| %.3e (reference uses FP16-rounded activations)%s\n",
                    name, N, M, K, nmse, max_rel, nmse <= 1e-9 && max_rel <= 1e-5 ? "" : "  GATE FAILED");
        if (nan_missing || nan_unexpected || !(nmse <= 1e-9) || !(max_rel <= 1e-5)) {
            rc = 1;
        }
    };

    if (!bench) {
        const Var & v = *sel[0];
        CHECK(hipMemset(dd, 0, (size_t) Np * M * 4));
        conv();
        v.fn(ctx, v.splits);
        check(v.name);
        const Timing t = time_fn([&] { if (!v.fused) { conv(); } v.fn(ctx, v.splits); }, reps);
        std::printf("median %.1f us (min %.1f, p90 %.1f), %.2f TFLOPS (2*M*N*K); batch mean %.1f us\n", t.med_us, t.min_us,
                    t.p90_us, tflops(M, N, K, t.med_us), t.batch_us);
        return rc;
    }

    // hipBLAS baselines on the same shape: FP16 weights/activations, as llama.cpp's F16 path on RDNA 3 feeds it.
    {
        std::vector<_Float16> h16((size_t) M * K);
        std::normal_distribution<float> wd(0.0f, 0.05f);
        for (auto & h : h16) {
            h = (_Float16) wd(rng);
        }
        _Float16 *w16, *y16, *c16;
        float * c32;
        CHECK(hipMalloc(&w16, h16.size() * 2));
        CHECK(hipMalloc(&y16, (size_t) N * K * 2));
        CHECK(hipMalloc(&c16, (size_t) N * M * 2));
        CHECK(hipMalloc(&c32, (size_t) N * M * 4));
        CHECK(hipMemcpy(w16, h16.data(), h16.size() * 2, hipMemcpyHostToDevice));
        std::vector<_Float16> hy16((size_t) N * K);
        for (size_t i = 0; i < hy16.size(); ++i) {
            hy16[i] = (_Float16) hy[i];
        }
        CHECK(hipMemcpy(y16, hy16.data(), hy16.size() * 2, hipMemcpyHostToDevice));
        hipblasHandle_t h;
        if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "hipblasCreate failed\n");
            return 1;
        }
        const __half a16 = __float2half(1.0f), b16 = __float2half(0.0f);
        const float a32 = 1.0f, b32 = 0.0f;
        auto g16 = [&] {
            hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &a16, w16, HIPBLAS_R_16F, K, y16, HIPBLAS_R_16F, K, &b16, c16,
                          HIPBLAS_R_16F, M, HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT);
        };
        auto g32 = [&] {
            hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &a32, w16, HIPBLAS_R_16F, K, y16, HIPBLAS_R_16F, K, &b32, c32,
                          HIPBLAS_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        };
        const Timing t16 = time_fn(g16, reps), t32 = time_fn(g32, reps);
        std::printf("RESULT M=%d N=%d K=%d hipblas16F %.1f us %.2f TF (batch %.1f) | hipblas32F %.1f us %.2f TF (batch %.1f)\n", M, N, K,
                    t16.med_us, tflops(M, N, K, t16.med_us), t16.batch_us, t32.med_us, tflops(M, N, K, t32.med_us), t32.batch_us);
        CHECK(hipFree(w16));
        CHECK(hipFree(y16));
        CHECK(hipFree(c16));
        CHECK(hipFree(c32));
        hipblasDestroy(h);
    }
    const Timing tc = time_fn(conv, reps);
    std::printf("RESULT M=%d N=%d K=%d act-convert %.1f us\n", M, N, K, tc.med_us);
    for (const Var * v : sel) {
        CHECK(hipMemset(dd, 0, (size_t) Np * M * 4));
        conv();
        v->fn(ctx, v->splits);
        check(v->name);
        const Timing tg = time_fn([&] { v->fn(ctx, v->splits); }, reps);
        const Timing tt = time_fn([&] { if (!v->fused) { conv(); } v->fn(ctx, v->splits); }, reps);
        std::printf("RESULT M=%d N=%d K=%d %s gemm %.1f us %.2f TF (batch %.1f) | conv+gemm %.1f us %.2f TF (batch %.1f)\n", M, N, K, v->name,
                    tg.med_us, tflops(M, N, K, tg.med_us), tg.batch_us, tt.med_us, tflops(M, N, K, tt.med_us), tt.batch_us);
    }
    return rc;
}
