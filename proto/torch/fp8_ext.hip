// C entry point to the decode prototype for ctypes (proto/torch/fp8_linear.py).
#define FP8_GEMV_NO_MAIN
#include "../gemv/fp8_gemv.hip"

// yh: N*K/2 uint32 scratch (variant b), ovf: one int scratch (variant b). Returns a hipError_t value.
extern "C" int fp8_gemv_run(char variant, int R, int N, const void * w, const void * y, void * yh, void * ovf, void * dst,
                            int M, int K, float out_scale, void * stream) {
    if (K % 16 != 0 || N < 1 || N > 8 || (variant != 'a' && variant != 'b' && variant != 'c')) {
        return (int) hipErrorInvalidValue;
    }
    const Buffers b{(const uint8_t *) w, (const float *) y, (uint32_t *) yh, (int *) ovf, (float *) dst};
    const hipStream_t st = (hipStream_t) stream;
    switch (R) {
        case 1: launch_n<1>(N, variant, b, M, K, out_scale, st); break;
        case 2: launch_n<2>(N, variant, b, M, K, out_scale, st); break;
        case 4: launch_n<4>(N, variant, b, M, K, out_scale, st); break;
        default: return (int) hipErrorInvalidValue;
    }
    return (int) hipGetLastError();
}
