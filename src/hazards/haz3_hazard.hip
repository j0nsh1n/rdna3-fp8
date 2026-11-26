// Case 1 on gfx1100: a matrix result used as the next matrix instruction's A, across a call that is not inlined.
// The RDNA 4 study (measurements/compiler-and-hazards.md, src/study/haz_hazard.hip) found that on gfx1201 clang puts no
// V_NOP between a callee's last matrix instruction and the caller's next one. This is the same set of shapes with the
// gfx1100 builtins (16-half A/B, 8-register D), each against an exact CPU answer.
// Usage: haz3_hazard [launches]    (default 8; every kernel is launched that many times, D preset to a sentinel)
#include "haz3_common.h"

using namespace haz3;

#define WMMA16(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32((a), (b), (c), false)
#define WMMA32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))

// Reference kernel shape: straight line, D1 (16-bit result) is the next instruction's A.
__global__ void h_straight(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h d1 = WMMA16(A[l], B[l], C16[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// Call boundary: the first matrix instruction is the last thing in a function that is not inlined.
__device__ __attribute__((noinline)) v16h first_mm(v16h a, v16h b, v16h c) { return WMMA16(a, b, c); }
__global__ void h_call(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h d1 = first_mm(A[l], B[l], C16[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// Same, with the result left in the return registers (C is the first argument, so D needs no copy).
__device__ __attribute__((noinline)) v16h first_mm_tied(v16h c, v16h a, v16h b) { return WMMA16(a, b, c); }
__global__ void h_call2(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// As h_call2, second instruction's other operands loaded before the call; an empty asm keeps them in registers.
__global__ void h_call3(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v16h b2 = B2[l];
    v8f c2 = C2[l];
    asm volatile("" : "+v"(b2), "+v"(c2));
    const v16h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, b2, c2);
}

// The same in plain C++ (operands read before the call).
__global__ void h_call4(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h b2 = B2[l];
    const v8f c2 = C2[l];
    const v16h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, b2, c2);
}

// Inline assembly: the dependent pair with a V_NOP between (what the RDNA 4 fix inserts) and without it (the break).
__global__ void h_asm_nop(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v16h d1 = C16[l];
    v8f d2;
    asm volatile(
        "v_wmma_f16_16x16x16_f16 %0, %2, %3, %0\n\t"
        "v_nop\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %0, %4, %5"
        : "+&v"(d1), "=&v"(d2)
        : "v"(A[l]), "v"(B[l]), "v"(B2[l]), "v"(C2[l]));
    D[l] = d2;
}
__global__ void h_asm_raw(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v16h d1 = C16[l];
    v8f d2;
    asm volatile(
        "v_wmma_f16_16x16x16_f16 %0, %2, %3, %0\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %0, %4, %5"
        : "+&v"(d1), "=&v"(d2)
        : "v"(A[l]), "v"(B[l]), "v"(B2[l]), "v"(C2[l]));
    D[l] = d2;
}

// Positive control for the probe itself: the same pair with D1's register replaced by the stale one (C16) as A.
// It must differ from the model; if it did not, the model could not tell a stale read from a right one.
__global__ void h_asm_stale(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v16h d1 = C16[l];
    v8f d2;
    asm volatile(
        "v_wmma_f16_16x16x16_f16 %0, %2, %3, %0\n\t"
        "v_nop\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %6, %4, %5"
        : "+&v"(d1), "=&v"(d2)
        : "v"(A[l]), "v"(B[l]), "v"(B2[l]), "v"(C2[l]), "v"(C16[l]));
    D[l] = d2;
}

namespace {

const float SENT = __builtin_bit_cast(float, 0x7FBADBADu);

using K = void (*)(const v16h *, const v16h *, const v16h *, const v16h *, const v8f *, v8f *);

}  // namespace

int main(int argc, char ** argv) {
    const int launches = argc > 1 ? std::atoi(argv[1]) : 8;
    require_arch();
    Verdicts v;
    const Problem p1 = make_problem(3), p2 = make_problem(5);

    // Stage 1: 16-bit accumulator. c16 even halves hold small integers, odd halves zero (finite whatever the hardware
    // leaves there). B2 has odd k rows zero, so the unused odd halves of D1 never enter stage 2.
    std::vector<uint16_t> c16(32 * 16, 0), b2(32 * 16, 0);
    Rng rc(9);
    for (int l = 0; l < 32; ++l) {
        for (int e = 0; e < 8; ++e) {
            c16[l * 16 + 2 * e] = h16((float) rc.next());
        }
    }
    for (int l = 0; l < 16; ++l) {
        for (int k = 0; k < 16; k += 2) {
            b2[l * 16 + k] = b2[(l + 16) * 16 + k] = p2.b[l * 16 + k];
        }
    }
    double d1[16][16];  // D1[row][col]
    bool exact = true;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            const int lane = (i & 1) * 16 + j, e = i >> 1;
            double s = h2f(c16[lane * 16 + 2 * e]);
            for (int k = 0; k < 16; ++k) {
                s += (double) h2f(p1.a[i * 16 + k]) * (double) h2f(p1.b[j * 16 + k]);
            }
            exact = exact && std::fabs(s) < 2048 && s == std::floor(s);
            d1[i][j] = s;
        }
    }
    v.put("hazard_ref_exact", exact, "stage-1 values are small integers, exact in FP16");
    // Stage 2: each 16-lane half of the wave reads its own copy of A (found with the first run: the replicas in
    // lanes 16-31 are not ignored, and here the two halves hold different data). Output row i = 2e + g (g = lane
    // half) therefore uses A row i from lane 16g + i, whose element 2e' is D1[row 2e' + g][col i].
    std::vector<float> ref(32 * 8), stale(32 * 8);
    for (int l = 0; l < 32; ++l) {
        const int g = l >> 4;
        for (int e = 0; e < 8; ++e) {
            const int i = 2 * e + g, j = l & 15;
            double s = p2.c[l * 8 + e], st = p2.c[l * 8 + e];
            for (int k = 0; k < 16; k += 2) {
                s += d1[k + g][i] * (double) h2f(b2[j * 16 + k]);
                // Stale model: A2 holds C16's registers (the old contents of the result registers).
                st += (double) h2f(c16[(16 * g + i) * 16 + k]) * (double) h2f(b2[j * 16 + k]);
            }
            ref[l * 8 + e] = (float) s;
            stale[l * 8 + e] = (float) st;
        }
    }

    v16h * dA = (v16h *) dev_copy(p1.a);
    v16h * dB = (v16h *) dev_copy(p1.b);
    v16h * dC16 = (v16h *) dev_copy(c16);
    v16h * dB2 = (v16h *) dev_copy(b2);
    v8f * dC2 = (v8f *) dev_copy(p2.c);
    v8f * dD;
    CHECK(hipMalloc(&dD, 32 * 8 * 4));

    struct Case {
        const char * name;
        K k;
        bool must_be_right;  // false for the deliberate break
    } cases[] = {{"straight_builtin", h_straight, true}, {"call_plain", h_call, true},  {"call_tied", h_call2, true},
                 {"call_tied_asm_order", h_call3, true}, {"call_tied_plain_cpp", h_call4, true},
                 {"asm_with_vnop", h_asm_nop, true},     {"asm_no_vnop_BREAK", h_asm_raw, false},
                 {"asm_stale_A_CONTROL", h_asm_stale, false}};
    std::vector<float> first_raw;
    for (const Case & c : cases) {
        std::string counts;
        bool any_wrong = false, same = true, near_stale = true;
        std::vector<float> first;
        for (int n = 0; n < launches; ++n) {
            std::vector<float> s(32 * 8, SENT);
            CHECK(hipMemcpy(dD, s.data(), 32 * 8 * 4, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(c.k, dim3(1), dim3(32), 0, 0, dA, dB, dC16, dB2, dC2, dD);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const auto d = host_copy((const float *) dD, 32 * 8);
            const int w = diff(d, ref);
            any_wrong |= w > 0;
            counts += (n ? "," : "") + std::to_string(w);
            if (n == 0) {
                first = d;
            } else {
                same = same && diff(d, first) == 0;
            }
            near_stale = near_stale && diff(d, stale) == 0;
        }
        if (std::strcmp(c.name, "asm_no_vnop_BREAK") == 0) {
            first_raw = first;
        }
        v.put(c.name, !any_wrong,
              fmt("wrong results per launch of 256: [%s]; repeat identical: %s; equals the stale-A prediction: %s%s",
                  counts.c_str(), same ? "yes" : "no", near_stale ? "yes" : "no",
                  c.must_be_right ? "" : " (control: FAIL expected where the hazard exists)"));
    }
    // Equality of each call-shape result with the no-V_NOP assembly result (RDNA 4: bit for bit).
    for (const Case & c : cases) {
        if (std::strncmp(c.name, "call_", 5) != 0 && std::strncmp(c.name, "straight", 8) != 0) {
            continue;
        }
        std::vector<float> s(32 * 8, SENT);
        CHECK(hipMemcpy(dD, s.data(), 32 * 8 * 4, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(c.k, dim3(1), dim3(32), 0, 0, dA, dB, dC16, dB2, dC2, dD);
        CHECK(hipDeviceSynchronize());
        const auto d = host_copy((const float *) dD, 32 * 8);
        v.info("equals_no_vnop_asm", fmt("%s: %d of 256 differ from the no-V_NOP inline-assembly result", c.name,
                                         diff(d, first_raw)));
    }
    return v.fails ? 1 : 0;
}
