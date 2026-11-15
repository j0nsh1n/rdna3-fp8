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
// Usage: fp8_gemm <N> [M K reps]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32;
    const int wm0 = (wave / 4) * WM;  // 2 waves along M
    const int wn0 = (wave % 4) * WN;  // 4 waves along N

    float8_t acc[TM][TN] = {};

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

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + wn0 + j * 16 + lane % 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + wm0 + i * 16 + 2 * e + lane / 16;
                if (n < N) {
                    dst[(size_t) n * M + m] = acc[i][j][e] * out_scale;
                }
            }
        }
    }
}

static float e4m3_to_float(uint8_t x) {
    const int ax = x & 0x7F;
    const int e = ax >> 3, m = ax & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return x & 0x80 ? -v : v;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <N> [M K reps]\n", argv[0]);
        return 2;
    }
    const int N = std::atoi(argv[1]);
    const int M = argc > 2 ? std::atoi(argv[2]) : 16384;
    const int K = argc > 3 ? std::atoi(argv[3]) : 14336;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 50;
    if (M % BM != 0 || K % BK != 0 || N < 1) {
        std::fprintf(stderr, "need M %% %d == 0, K %% %d == 0, N >= 1\n", BM, BK);
        return 2;
    }
    const int Np = (N + BN - 1) / BN * BN;
    const float tensor_scale = 0.0123f;
    const float out_scale = tensor_scale * 256.0f;

    std::mt19937 rng(1234);
    std::vector<uint8_t> hw((size_t) M * K);
    for (auto & b : hw) {
        do {
            b = (uint8_t) rng();
        } while ((b & 0x7F) == 0x7F);
    }
    std::vector<float> hy((size_t) Np * K, 0.0f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t) N * K; ++i) {
        hy[i] = nd(rng);
    }

    uint8_t * dw;
    float *dy, *dd;
    uint32_t * dyh;
    CHECK(hipMalloc(&dw, hw.size()));
    CHECK(hipMalloc(&dy, hy.size() * 4));
    CHECK(hipMalloc(&dyh, hy.size() * 2));
    CHECK(hipMalloc(&dd, (size_t) Np * M * 4));
    CHECK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dy, hy.data(), hy.size() * 4, hipMemcpyHostToDevice));

    const size_t words = (size_t) Np * K / 4;
    auto run = [&] {
        act_to_f16_pairs<<<(words + 255) / 256, 256>>>(dy, dyh, words);
        gemm_fp8<<<dim3(M / BM, Np / BN), THREADS>>>(dw, dyh, dd, M, N, K, out_scale);
    };
    run();
    CHECK(hipDeviceSynchronize());
    std::vector<float> got((size_t) N * M);
    CHECK(hipMemcpy(got.data(), dd, got.size() * 4, hipMemcpyDeviceToHost));

    // Reference on a sample of outputs (full CPU reference at N=4096 is ~1e12 products).
    float lut[256];
    for (int c = 0; c < 256; ++c) {
        lut[c] = e4m3_to_float((uint8_t) c);
    }
    double err2 = 0, ref2 = 0, max_rel = 0;
    std::uniform_int_distribution<int> pm(0, M - 1), pn(0, N - 1);
    const int samples = 4096;
    for (int s = 0; s < samples; ++s) {
        // First outputs of every tile row and column exhaustively in the first 256, then random ones.
        const int m = s < 256 ? s % M : pm(rng);
        const int n = s < 256 ? (s / 16) % N : pn(rng);
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
    std::printf("N=%d M=%d K=%d: sampled NMSE %.3e, max |err|/sum|w*a| %.3e (reference uses FP16-rounded activations)\n",
                N, M, K, err2 / ref2, max_rel);

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int i = 0; i < 3; ++i) {
        run();
    }
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
        CHECK(hipEventRecord(t0));
        run();
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float m;
        CHECK(hipEventElapsedTime(&m, t0, t1));
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("median %.1f us (min %.1f, p90 %.1f), %.2f TFLOPS (2*M*N*K)\n", med * 1e3, ms.front() * 1e3,
                ms[ms.size() * 9 / 10] * 1e3, 2.0 * M * N * (double) K / (med * 1e9));
    return 0;
}
