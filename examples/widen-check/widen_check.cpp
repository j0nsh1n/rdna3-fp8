// CPU check of the in-register FP8 E4M3 -> FP16 widening planned for the GPU kernels.
// One 32-bit word holds weights b0..b3 (b0 lowest). Two half2 come out:
//   even = (b0, b2), odd = (b1, b3), each FP16 equal to the E4M3 value times 2^-8.
// Compared with llama.cpp's own CPU conversion for every possible word, plus the per-word NaN-code check.
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint32_t widen_even(uint32_t x) {
    return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3F803F80u);
}

static uint32_t widen_odd(uint32_t x) {
    return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u);
}

// Non-zero exactly when some byte of x is an E4M3 NaN code (low seven bits all ones: 0x7F or 0xFF).
static uint32_t nan_bytes(uint32_t x) {
    return ((x & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
}

static float half_bits_to_float(uint16_t h) {
    _Float16 f;
    std::memcpy(&f, &h, 2);
    return (float) f;
}

int main() {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    float ref[256];
    uint8_t codes[256];
    for (int c = 0; c < 256; ++c) {
        codes[c] = (uint8_t) c;
    }
    traits->to_float(codes, ref, 256);

    // Per-code table first, so a failure names the code.
    int bad_codes = 0, nan_codes = 0;
    for (int c = 0; c < 256; ++c) {
        const uint32_t x = (uint32_t) c;
        const float w = half_bits_to_float((uint16_t) widen_even(x)) * 256.0f;
        if (std::isnan(ref[c])) {
            ++nan_codes;
            std::printf("code 0x%02X: llama.cpp NaN, widened %g\n", c, w);
            continue;
        }
        if (w != ref[c] || std::signbit(w) != std::signbit(ref[c])) {
            ++bad_codes;
            std::printf("MISMATCH code 0x%02X: llama.cpp %g, widened %g\n", c, ref[c], w);
        }
    }
    std::printf("single codes: %d mismatches, %d NaN codes\n", bad_codes, nan_codes);

    // Every 32-bit word: catches one byte leaking into its neighbour's half.
    uint64_t bad_words = 0, bad_nan = 0;
    for (uint64_t xx = 0; xx <= 0xFFFFFFFFull; ++xx) {
        const uint32_t x = (uint32_t) xx;
        const uint32_t e = widen_even(x), o = widen_odd(x);
        const uint16_t h[4] = {(uint16_t) e, (uint16_t) o, (uint16_t) (e >> 16), (uint16_t) (o >> 16)};
        bool any_nan = false;
        for (int i = 0; i < 4; ++i) {
            any_nan |= ((x >> (8 * i)) & 0x7F) == 0x7F;
        }
        if ((nan_bytes(x) != 0) != any_nan && ++bad_nan <= 5) {
            std::printf("NAN-CHECK MISMATCH word 0x%08X: expected %d\n", x, (int) any_nan);
        }
        for (int i = 0; i < 4; ++i) {
            const uint8_t b = (uint8_t) (x >> (8 * i));
            if ((b & 0x7F) == 0x7F) {
                continue;
            }
            const float w = half_bits_to_float(h[i]) * 256.0f;
            if (w != ref[b] || std::signbit(w) != std::signbit(ref[b])) {
                if (++bad_words <= 5) {
                    std::printf("MISMATCH word 0x%08X byte %d: llama.cpp %g, widened %g\n", x, i, ref[b], w);
                }
            }
        }
    }
    std::printf("all 2^32 words: %llu mismatching bytes (NaN codes skipped), %llu NaN-check mismatches\n",
                (unsigned long long) bad_words, (unsigned long long) bad_nan);
    return bad_codes == 0 && bad_words == 0 && bad_nan == 0 ? 0 : 1;
}
