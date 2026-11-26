// RDNA 3 (gfx1100) port of the RDNA 4 study's compiler-hazard probes. Self-contained: HIP runtime, integer CPU model.
//
// gfx1100 register layout of v_wmma_*_16x16x16_f16_w32 (README "Matrix arithmetic", MEASURED there):
//   A row r:    lane r (and the copy in lane r + 16), 16 halves k = 0..15 in the lane's 16-half vector.
//   B column c: lane c (and lane c + 16), half k.
//   D / C (f32): lane l, float e (0..7) = element (row 2*e + (l >> 4), column l & 15).
//   D / C (f16, opsel = 0): lane l, VGPR e, low half = vector element 2*e = the same (row, column); odd elements unused.
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef HARNESS_ARCH
#define HARNESS_ARCH "gfx1100"
#endif

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

namespace haz3 {

inline void require_arch() {
    int n = 0;
    CHECK(hipGetDeviceCount(&n));
    if (n < 1) {
        std::fprintf(stderr, "no HIP device\n");
        std::exit(3);
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, HARNESS_ARCH, std::strlen(HARNESS_ARCH)) != 0) {
        std::fprintf(stderr, "device 0 is %s (%s), built for %s; refusing to run\n", prop.name, prop.gcnArchName,
                     HARNESS_ARCH);
        std::exit(3);
    }
}

inline uint16_t h16(float f) {
    const _Float16 h = (_Float16) f;
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
inline float h2f(uint16_t u) {
    _Float16 h;
    std::memcpy(&h, &u, 2);
    return (float) h;
}
inline uint32_t f32b(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// Small NON-NEGATIVE integers in [0, 3]. Not [-3, 3]: v_wmma_f32_16x16x16_f16 is off by a few units of 2^(E-24) when the
// terms have mixed signs (README, measurements/matrix-arithmetic.md), so a signed integer probe fails on a good run.
// With all terms >= 0 every product and partial sum is exact in FP16 and FP32, so the CPU answer is exact.
struct Rng {
    uint32_t x;
    explicit Rng(uint32_t seed) : x(seed * 2654435761u + 12345u) {}
    int next() {
        x = x * 1664525u + 1013904223u;
        return (int) ((x >> 16) % 4);
    }
};

// Per-lane images. a, b: 32 lanes x 16 halves (lanes 16-31 repeat lanes 0-15). c: 32 lanes x 8 floats.
struct Problem {
    std::vector<uint16_t> a, b;
    std::vector<float> c;
    Problem() : a(32 * 16), b(32 * 16), c(32 * 8) {}
};

inline Problem make_problem(uint32_t seed) {
    Rng r(seed);
    Problem p;
    for (int l = 0; l < 16; ++l) {
        for (int k = 0; k < 16; ++k) {
            p.a[l * 16 + k] = p.a[(l + 16) * 16 + k] = h16((float) r.next());
            p.b[l * 16 + k] = p.b[(l + 16) * 16 + k] = h16((float) r.next());
        }
    }
    for (int i = 0; i < 32 * 8; ++i) {
        p.c[i] = (float) r.next();
    }
    return p;
}

// D = A * B + C as f32 results in the D lane layout (exact, in double).
inline std::vector<float> reference(const Problem & p) {
    std::vector<float> d(32 * 8);
    for (int l = 0; l < 32; ++l) {
        for (int e = 0; e < 8; ++e) {
            const int row = 2 * e + (l >> 4), col = l & 15;
            double s = p.c[l * 8 + e];
            for (int k = 0; k < 16; ++k) {
                s += (double) h2f(p.a[row * 16 + k]) * (double) h2f(p.b[col * 16 + k]);
            }
            d[l * 8 + e] = (float) s;
        }
    }
    return d;
}

template <class T>
T * dev_copy(const std::vector<T> & h) {
    T * d = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> host_copy(const T * d, size_t n) {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

inline std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char * f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// Machine-readable result line: RESULT <name> <PASS|FAIL|INFO> <text>.
struct Verdicts {
    int fails = 0;
    void put(const char * name, bool ok, const std::string & text) {
        std::printf("RESULT %s %s %s\n", name, ok ? "PASS" : "FAIL", text.c_str());
        fails += !ok;
    }
    void info(const char * name, const std::string & text) { std::printf("RESULT %s INFO %s\n", name, text.c_str()); }
};

inline int diff(const std::vector<float> & x, const std::vector<float> & y) {
    int w = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        w += f32b(x[i]) != f32b(y[i]);
    }
    return w;
}

}  // namespace haz3
