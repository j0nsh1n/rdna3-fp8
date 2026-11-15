// Device memory bandwidth on one GPU: a read-only pass (what a decode GEMV does with its weights) and a copy.
// Both use 16-byte loads in a grid-stride loop. Prints the median and best of the timed repetitions.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                              \
    do {                                                                                      \
        hipError_t e_ = (x);                                                                  \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

__global__ void read_kernel(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // The data is zero, so this never writes; it keeps the loads from being removed.
    if (acc == 0xdeadbeefu) {
        *sink = acc;
    }
}

__global__ void copy_kernel(const uint4 * __restrict__ src, uint4 * __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static void report(const char * name, std::vector<float> ms, double bytes) {
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const double best = ms.front();
    std::printf("%-5s bytes moved %.0f MiB  median %.3f ms = %.1f GB/s  best %.3f ms = %.1f GB/s\n", name,
                bytes / (1 << 20), med, bytes / (med * 1e6), best, bytes / (best * 1e6));
}

int main(int argc, char ** argv) {
    const size_t bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024ull) << 20;  // MiB per buffer
    const int reps = 50;
    const size_t n = bytes / sizeof(uint4);

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s (%s), %d CUs, buffer %zu MiB, %d reps\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount, bytes >> 20, reps);

    uint4 *src, *dst;
    unsigned * sink;
    CHECK(hipMalloc(&src, bytes));
    CHECK(hipMalloc(&dst, bytes));
    CHECK(hipMalloc(&sink, sizeof(unsigned)));
    CHECK(hipMemset(src, 0, bytes));
    CHECK(hipMemset(dst, 0, bytes));

    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));

    auto time_it = [&](auto launch) {
        for (int i = 0; i < 5; ++i) {
            launch();
        }
        CHECK(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int i = 0; i < reps; ++i) {
            CHECK(hipEventRecord(t0));
            launch();
            CHECK(hipEventRecord(t1));
            CHECK(hipEventSynchronize(t1));
            float m;
            CHECK(hipEventElapsedTime(&m, t0, t1));
            ms.push_back(m);
        }
        return ms;
    };

    report("read", time_it([&] { read_kernel<<<grid, block>>>(src, n, sink); }), (double) bytes);
    report("copy", time_it([&] { copy_kernel<<<grid, block>>>(src, dst, n); }), 2.0 * bytes);

    CHECK(hipFree(src));
    CHECK(hipFree(dst));
    CHECK(hipFree(sink));
    return 0;
}
