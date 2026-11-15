// Throwaway prototype: FP8 E4M3 decode GEMV for gfx1100 (RX 7900 XT).
// Weights stay as bytes [M][K] (row-major, K contiguous, like llama.cpp), activations F32 [N][K], output F32 [N][M].
// A weight word of four bytes b0..b3 widens to two half2 by bit shifts: even = (b0, b2), odd = (b1, b3), each half
// equal to the E4M3 value times 2^-8 (checked on the CPU by proto/widen_check.cpp). The 2^8 goes into the scale.
//
// Variant A (exact): FP32 activations, fma(float(half w), a, acc) -> v_fma_mix_f32. Same products as llama.cpp.
// Variant B (f16dot): activations converted to FP16 once per call, v_dot2_f32_f16. Half the multiply instructions;
//                     rounds activations to FP16.
// R = rows per wave (activation reuse in registers). N = batch columns (1..8).
//
// Usage: fp8_gemv <variant a|b> <R 1|2|4> <N> [M K reps]
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

constexpr int WAVE = 32;
constexpr int WAVES_PER_BLOCK = 4;

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint4 load16_nt(const uint8_t * row, int c) {
    const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(row) + c);
    return make_uint4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ uint32_t widen_even(uint32_t x) {
    return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3F803F80u);
}

__device__ __forceinline__ uint32_t widen_odd(uint32_t x) {
    return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u);
}

__device__ __forceinline__ float lo_f(uint32_t h2) {
    return __half2float(__ushort_as_half((unsigned short) (h2 & 0xFFFF)));
}

__device__ __forceinline__ float hi_f(uint32_t h2) {
    return __half2float(__ushort_as_half((unsigned short) (h2 >> 16)));
}

template <int N, int R>
__device__ __forceinline__ void wave_reduce_store(float (&acc)[N][R], int row0, int M, float out_scale, float * dst) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[n][r];
#pragma unroll
            for (int off = WAVE / 2; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, WAVE);
            }
            acc[n][r] = v;
        }
    }
    const int lane = threadIdx.x % WAVE;
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (lane == n * R + r && row0 + r < M) {
                dst[(size_t) n * M + row0 + r] = acc[n][r] * out_scale;
            }
        }
    }
}

// Variant A: 16 weight bytes per lane per step, FP32 activations.
template <int N, int R>
__global__ __launch_bounds__(WAVE * WAVES_PER_BLOCK) void gemv_a(const uint8_t * __restrict__ w, const float * __restrict__ y,
                                                                float * __restrict__ dst, int M, int K, float out_scale) {
    const int wave = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row0 = wave * R;
    if (row0 >= M) {
        return;
    }
    float acc[N][R] = {};
    const int k16 = K / 16;
    for (int c = lane; c < k16; c += WAVE) {
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(row0 + r, M - 1);
            wv[r] = load16_nt(w + (size_t) row * K, c);
        }
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const float4 * yp = reinterpret_cast<const float4 *>(y + (size_t) n * K) + c * 4;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 a = yp[q];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t word = q == 0 ? wv[r].x : q == 1 ? wv[r].y : q == 2 ? wv[r].z : wv[r].w;
                    const uint32_t e = widen_even(word);
                    const uint32_t o = widen_odd(word);
                    float s = acc[n][r];
                    s = fmaf(lo_f(e), a.x, s);
                    s = fmaf(lo_f(o), a.y, s);
                    s = fmaf(hi_f(e), a.z, s);
                    s = fmaf(hi_f(o), a.w, s);
                    acc[n][r] = s;
                }
            }
        }
    }
    wave_reduce_store<N, R>(acc, row0, M, out_scale, dst);
}

// Variant B pre-pass: F32 activations -> FP16, pairs reordered to match (b0,b2),(b1,b3) per weight word.
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

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ half2_t as_h2(uint32_t v) {
    half2_t h;
    std::memcpy(&h, &v, 4);
    return h;
}

// Variant B: 16 weight bytes per lane per step, FP16 activation pairs (8 uint32 per 16 weights).
template <int N, int R>
__global__ __launch_bounds__(WAVE * WAVES_PER_BLOCK) void gemv_b(const uint8_t * __restrict__ w, const uint32_t * __restrict__ yh,
                                                                float * __restrict__ dst, int M, int K, float out_scale) {
    const int wave = blockIdx.x * WAVES_PER_BLOCK + threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row0 = wave * R;
    if (row0 >= M) {
        return;
    }
    float acc[N][R] = {};
    const int k16 = K / 16;
    for (int c = lane; c < k16; c += WAVE) {
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(row0 + r, M - 1);
            wv[r] = load16_nt(w + (size_t) row * K, c);
        }
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const uint4 * yp = reinterpret_cast<const uint4 *>(yh + (size_t) n * (K / 2)) + c * 2;
            const uint4 y0 = yp[0], y1 = yp[1];
            const uint32_t ya[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t words[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
                float s = acc[n][r];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    // The empty asm keeps each widened word whole; without it LLVM splits the masks into 16-bit ops.
                    uint32_t e = widen_even(words[q]), o = widen_odd(words[q]);
                    asm("" : "+v"(e), "+v"(o));
                    s = __builtin_amdgcn_fdot2(as_h2(e), as_h2(ya[2 * q]), s, false);
                    s = __builtin_amdgcn_fdot2(as_h2(o), as_h2(ya[2 * q + 1]), s, false);
                }
                acc[n][r] = s;
            }
        }
    }
    wave_reduce_store<N, R>(acc, row0, M, out_scale, dst);
}

static float e4m3_to_float(uint8_t x) {
    const int ax = x & 0x7F;
    const int e = ax >> 3, m = ax & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return x & 0x80 ? -v : v;
}

template <int N, int R>
static void launch(char variant, const uint8_t * w, const float * y, uint32_t * yh, float * dst, int M, int K, float s) {
    const int waves = (M + R - 1) / R;
    const int blocks = (waves + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
    if (variant == 'a') {
        gemv_a<N, R><<<blocks, WAVE * WAVES_PER_BLOCK>>>(w, y, dst, M, K, s);
    } else {
        const size_t words = (size_t) N * K / 4;
        act_to_f16_pairs<<<(words + 255) / 256, 256>>>(y, yh, words);
        gemv_b<N, R><<<blocks, WAVE * WAVES_PER_BLOCK>>>(w, yh, dst, M, K, s);
    }
}

template <int R>
static void launch_n(int N, char v, const uint8_t * w, const float * y, uint32_t * yh, float * d, int M, int K, float s) {
    switch (N) {
        case 1: launch<1, R>(v, w, y, yh, d, M, K, s); break;
        case 2: launch<2, R>(v, w, y, yh, d, M, K, s); break;
        case 4: launch<4, R>(v, w, y, yh, d, M, K, s); break;
        case 8: launch<8, R>(v, w, y, yh, d, M, K, s); break;
        default: std::fprintf(stderr, "N must be 1, 2, 4 or 8\n"); std::exit(2);
    }
}

static void require_gfx1100() {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1100", 7) != 0) {
        std::fprintf(stderr, "device 0 is %s (%s), not the gfx1100 card; refusing to time the wrong GPU\n", prop.name, prop.gcnArchName);
        std::exit(3);
    }
}

int main(int argc, char ** argv) {
    require_gfx1100();
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <a|b> <R 1|2|4> <N 1|2|4|8> [M K reps]\n", argv[0]);
        return 2;
    }
    const char variant = argv[1][0];
    const int R = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int M = argc > 4 ? std::atoi(argv[4]) : 16384;
    const int K = argc > 5 ? std::atoi(argv[5]) : 14336;
    const int reps = argc > 6 ? std::atoi(argv[6]) : 200;
    if (K % 16 != 0 || (variant != 'a' && variant != 'b')) {
        std::fprintf(stderr, "K must be a multiple of 16; variant a or b\n");
        return 2;
    }
    const float tensor_scale = 0.0123f;
    const float out_scale = tensor_scale * 256.0f;

    std::mt19937 rng(1234);
    std::vector<uint8_t> hw((size_t) M * K);
    for (auto & b : hw) {
        do {
            b = (uint8_t) rng();
        } while ((b & 0x7F) == 0x7F);
    }
    std::vector<float> hy((size_t) N * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : hy) {
        v = nd(rng);
    }

    uint8_t * dw;
    float *dy, *dd;
    uint32_t * dyh;
    CHECK(hipMalloc(&dw, hw.size()));
    CHECK(hipMalloc(&dy, hy.size() * 4));
    CHECK(hipMalloc(&dyh, hy.size() * 2));
    CHECK(hipMalloc(&dd, (size_t) N * M * 4));
    CHECK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dy, hy.data(), hy.size() * 4, hipMemcpyHostToDevice));

    auto run = [&] {
        switch (R) {
            case 1: launch_n<1>(N, variant, dw, dy, dyh, dd, M, K, out_scale); break;
            case 2: launch_n<2>(N, variant, dw, dy, dyh, dd, M, K, out_scale); break;
            case 4: launch_n<4>(N, variant, dw, dy, dyh, dd, M, K, out_scale); break;
            default: std::fprintf(stderr, "R must be 1, 2 or 4\n"); std::exit(2);
        }
    };

    run();
    CHECK(hipDeviceSynchronize());
    std::vector<float> got((size_t) N * M);
    CHECK(hipMemcpy(got.data(), dd, got.size() * 4, hipMemcpyDeviceToHost));

    double err2 = 0, ref2 = 0, max_rel = 0;
    float lut[256];
    for (int c = 0; c < 256; ++c) {
        lut[c] = e4m3_to_float((uint8_t) c);
    }
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double ref = 0, mag = 0;
            const uint8_t * wr = &hw[(size_t) m * K];
            const float * yr = &hy[(size_t) n * K];
            for (int k = 0; k < K; ++k) {
                ref += (double) lut[wr[k]] * yr[k];
                mag += std::fabs((double) lut[wr[k]] * yr[k]);
            }
            ref *= tensor_scale;
            mag *= tensor_scale;
            const double d = got[(size_t) n * M + m] - ref;
            err2 += d * d;
            ref2 += ref * ref;
            max_rel = std::max(max_rel, std::fabs(d) / mag);
        }
    }
    std::printf("variant %c R=%d N=%d M=%d K=%d: NMSE %.3e, max |err|/sum|w*a| %.3e\n", variant, R, N, M, K, err2 / ref2,
                max_rel);

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int i = 0; i < 10; ++i) {
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
    const double med = ms[ms.size() / 2] * 1e3;
    std::printf("median %.2f us (min %.2f, p90 %.2f), weight read %.1f GB/s\n", med, ms.front() * 1e3,
                ms[ms.size() * 9 / 10] * 1e3, (double) M * K / (med * 1e3));
    return 0;
}
