// Case 2 on gfx1100: a matrix builtin under a per-lane condition (if, early return, per-lane loop), and the safe forms.
// Port of src/study/haz_mask.hip of the RDNA 4 study. Natural HIP kernels; the listing scan says which run the matrix
// instruction under a partial EXEC, this program says what the card computes against an exact CPU answer.
// Usage: haz3_mask [launches]   (default 8)
#include "haz3_common.h"

using namespace haz3;

#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))

// Operands loaded by all lanes, result stored under a per-lane if (the shape the README says gfx11 once sank).
__global__ void k_sink(const v16h * A, const v16h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    const v16h a = A[l], b = B[l];
    const v8f c = C[l];
    const v8f d = WMMA(a, b, c);
    if (l < n) {
        D[l] = d;
    }
}

// Everything, loads included, under a per-lane if.
__global__ void k_ifload(const v16h * A, const v16h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    if (l < n) {
        D[l] = WMMA(A[l], B[l], C[l]);
    }
}

// Early return for lanes past the end of a partly filled tile.
__global__ void k_early(const v16h * A, const v16h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    if (l >= n) {
        return;
    }
    D[l] = WMMA(A[l], B[l], C[l]);
}

// Per-lane trip count through the accumulator.
__global__ void k_loop(const v16h * A, const v16h * B, const v8f * C, v8f * D, const int * trip) {
    const int l = threadIdx.x;
    v8f c = C[l];
    const int t = trip[l];
    for (int i = 0; i < t; ++i) {
        c = WMMA(A[i * 32 + l], B[i * 32 + l], c);
    }
    D[l] = c;
}

// Safe: padding by select, every lane runs the matrix instruction, store by all.
__global__ void k_select(const v16h * A, const v16h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    const bool ok = (l & 15) < n;
    const v16h z = {};
    const v16h a = ok ? A[l] : z;
    D[l] = WMMA(a, B[l], C[l]);
}

// Safe: early return on a wave-uniform bound (scalar branch).
__global__ void k_uniform(const v16h * A, const v16h * B, const v8f * C, v8f * D, int ntiles) {
    if ((int) blockIdx.x >= ntiles) {
        return;
    }
    const int l = threadIdx.x;
    D[l] = WMMA(A[l], B[l], C[l]);
}

// Hardware rule: EXEC set to `mask` around one matrix instruction in inline assembly; D preset to a sentinel.
__global__ void k_asm(const v16h * A, const v16h * B, const v8f * C, v8f * D, uint32_t mask) {
    const int l = threadIdx.x;
    const v16h a = A[l], b = B[l];
    const v8f c = C[l];
    v8f d = D[l];
    asm volatile(
        "s_mov_b32 s96, exec_lo\n\t"
        "s_mov_b32 exec_lo, %4\n\t"
        "v_wmma_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        "s_mov_b32 exec_lo, s96"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(c), "s"(mask)
        : "s96");
    D[l] = d;
}

namespace {

const float SENT = __builtin_bit_cast(float, 0x7FBADBADu);

struct Bufs {
    v16h *A, *B;
    v8f *C, *D;
};

// Wrong elements among lanes in `lanes` (bit mask), against ref.
int wrong_in(const std::vector<float> & d, const std::vector<float> & ref, uint32_t lanes) {
    int w = 0;
    for (int l = 0; l < 32; ++l) {
        if (!((lanes >> l) & 1)) {
            continue;
        }
        for (int e = 0; e < 8; ++e) {
            w += f32b(d[l * 8 + e]) != f32b(ref[l * 8 + e]);
        }
    }
    return w;
}

template <class Launch>
std::vector<float> run_once(Bufs & b, Launch launch) {
    std::vector<float> s(32 * 8, SENT);
    CHECK(hipMemcpy(b.D, s.data(), 32 * 8 * 4, hipMemcpyHostToDevice));
    launch();
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return host_copy((const float *) b.D, 32 * 8);
}

}  // namespace

int main(int argc, char ** argv) {
    const int launches = argc > 1 ? std::atoi(argv[1]) : 8;
    require_arch();
    Verdicts v;
    const Problem p = make_problem(7);
    const std::vector<float> ref = reference(p);
    Bufs b{};
    b.A = (v16h *) dev_copy(p.a);
    b.B = (v16h *) dev_copy(p.b);
    b.C = (v8f *) dev_copy(p.c);
    CHECK(hipMalloc(&b.D, 32 * 8 * 4));

    // Unsafe shapes with a lane mask lane < n. Valid output = the stored lanes; reference = all lanes' operands as in
    // memory (what the programmer of a partly filled tile means). FAIL = wrong results.
    struct K {
        const char * name;
        void (*k)(const v16h *, const v16h *, const v8f *, v8f *, int);
    } ks[] = {{"sink", k_sink}, {"ifload", k_ifload}, {"early", k_early}};
    for (const K & k : ks) {
        for (int n : {8, 16, 20, 31}) {
            const uint32_t lanes = (1u << n) - 1;
            std::string counts;
            bool any = false;
            for (int i = 0; i < launches; ++i) {
                const auto d = run_once(b, [&] { hipLaunchKernelGGL(k.k, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, n); });
                const int w = wrong_in(d, ref, lanes);
                any |= w > 0;
                counts += (i ? "," : "") + std::to_string(w);
            }
            v.put((std::string("unsafe_") + k.name).c_str(), !any,
                  fmt("n=%d: wrong results per launch of %d stored: [%s]", n, 8 * n, counts.c_str()));
        }
    }
    // Safe shapes: must be right.
    for (int n : {5, 16}) {
        Problem q = p;
        for (int l = 0; l < 32; ++l) {
            if ((l & 15) >= n) {
                for (int k = 0; k < 16; ++k) {
                    q.a[l * 16 + k] = 0;
                }
            }
        }
        // Rows with (row & 15) >= n are zero; A row = lane r, and lanes 16-31 repeat 0-15 in memory.
        const auto rq = reference(q);
        std::string counts;
        bool any = false;
        for (int i = 0; i < launches; ++i) {
            const auto d = run_once(b, [&] { hipLaunchKernelGGL(k_select, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, n); });
            const int w = wrong_in(d, rq, 0xFFFFFFFFu);
            any |= w > 0;
            counts += (i ? "," : "") + std::to_string(w);
        }
        v.put("safe_select", !any, fmt("rows >= %d padded by select: wrong per launch of 256: [%s]", n, counts.c_str()));
    }
    {
        std::string counts;
        bool any = false;
        for (int i = 0; i < launches; ++i) {
            const auto d = run_once(b, [&] { hipLaunchKernelGGL(k_uniform, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, 1); });
            const int w = wrong_in(d, ref, 0xFFFFFFFFu);
            any |= w > 0;
            counts += (i ? "," : "") + std::to_string(w);
        }
        v.put("safe_uniform", !any, fmt("wave-uniform early return: wrong per launch of 256: [%s]", counts.c_str()));
    }
    // Per-lane trip count: lanes 0-15 two trips, 16-31 one (unsafe); all lanes two trips (safe).
    {
        const Problem p2 = make_problem(11);
        std::vector<uint16_t> A2(32 * 16 * 2), B2(32 * 16 * 2);
        for (int i = 0; i < 32 * 16; ++i) {
            A2[i] = p.a[i];
            B2[i] = p.b[i];
            A2[32 * 16 + i] = p2.a[i];
            B2[32 * 16 + i] = p2.b[i];
        }
        v16h * dA = (v16h *) dev_copy(A2);
        v16h * dB = (v16h *) dev_copy(B2);
        Problem u = p2;
        u.c = ref;
        const auto ru = reference(u);
        for (int shape = 0; shape < 2; ++shape) {
            std::vector<int> trip(32);
            for (int l = 0; l < 32; ++l) {
                trip[l] = shape == 0 ? 2 : (l < 16 ? 2 : 1);
            }
            int * dt = dev_copy(trip);
            std::string counts;
            bool any = false;
            for (int i = 0; i < launches; ++i) {
                const auto d = run_once(b, [&] { hipLaunchKernelGGL(k_loop, dim3(1), dim3(32), 0, 0, dA, dB, b.C, b.D, dt); });
                // Reference: lanes with 2 trips = two-step model; lanes with 1 trip = first step only.
                int w = 0;
                for (int l = 0; l < 32; ++l) {
                    const auto & r = (shape == 0 || l < 16) ? ru : ref;
                    for (int e = 0; e < 8; ++e) {
                        w += f32b(d[l * 8 + e]) != f32b(r[l * 8 + e]);
                    }
                }
                any |= w > 0;
                counts += (i ? "," : "") + std::to_string(w);
            }
            v.put(shape == 0 ? "safe_loop_uniform" : "unsafe_loop_perlane", !any,
                  fmt("%s: wrong per launch of 256: [%s]", shape == 0 ? "all lanes 2 trips" : "lanes 0-15 two trips, 16-31 one",
                      counts.c_str()));
            CHECK(hipFree(dt));
        }
    }
    // Hardware rule on gfx1100 (inline assembly): which lanes' results are written with a partial EXEC, and are the
    // written values equal to the full-wave result.
    for (uint32_t m : {0xFFFFFFFFu, 0x0000FFFFu, 0xFFFF0000u, 0x55555555u, 0x00000001u}) {
        const auto d = run_once(b, [&] { hipLaunchKernelGGL(k_asm, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, m); });
        int off_written = 0, off_total = 0;
        for (int l = 0; l < 32; ++l) {
            for (int e = 0; ((m >> l) & 1) == 0 && e < 8; ++e) {
                ++off_total;
                off_written += f32b(d[l * 8 + e]) != f32b(SENT);
            }
        }
        v.info("asm_exec_rule", fmt("EXEC=0x%08x: %d of 256 results differ from the full-wave model; switched-off lanes' "
                                    "D written %d of %d",
                                    m, wrong_in(d, ref, 0xFFFFFFFFu), off_written, off_total));
    }
    return v.fails ? 1 : 0;
}
