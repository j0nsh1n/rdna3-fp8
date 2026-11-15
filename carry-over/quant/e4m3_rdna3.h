// Drop-in for clean/kernels/wg_platform.h on gfx11 (RDNA 3, no hardware FP8 conversion): software FP32 -> E4M3
// that is bit-exact with the repo's wg_f32_to_e4m3 (round to nearest even, saturating at +-448, NaN -> sign|0x7f,
// FP32 denormals -> signed zero), checked over all 2^32 inputs (RESULTS.md), plus the same rounding delivered as F16
// bits for the next F16 matrix operand. Uses WG_FN, u32, v2i, v8f, v8h from wg_platform.h.
//
// Method: with z = |x| clamped to [2^-6, 448] and C = 1.5 * 2^(E+20) (E = exponent of z), the FP32 add |x| + C rounds
// |x| to a multiple of 2^(E-3), ties to even, and the subtract is exact; 2^-6 as the floor gives the E4M3 subnormal
// step 2^-9. v_med3_f32 is used for clamps because fminf/fmaxf get a canonicalising v_max x,x in front on gfx11.
#pragma once

WG_FN float wg_q8m_clamp(float x, float lo, float hi) { return __builtin_amdgcn_fmed3f(x, lo, hi); }
WG_FN u32 wg_q8m_bits(float f) { return __builtin_bit_cast(u32, f); }
WG_FN float wg_q8m_flt(u32 u) { return __builtin_bit_cast(float, u); }

// C = 1.5 * 2^(E+20) for |x| (E clamped to [-6, 8]), as FP32 bits.
WG_FN u32 wg_q8m_magic(float ax) {
  return (wg_q8m_bits(wg_q8m_clamp(ax, 0.015625f, 448.0f)) & 0x7f800000u) + 0x0a400000u;
}

// ---- output (a): E4M3 code bytes -------------------------------------------------------------------------------
// The code comes straight from the rounded sum: bits(y) - bits(C) = k (grid index 8..16, or 0..8 below 2^-6) and
// code = 8 (E+6) + k = (bits(C) >> 20) - 1132 + k. min(., 0x7e) saturates (inf and, with NANFIX = false, NaN).
template <bool NANFIX>
WG_FN u32 wg_f32_to_e4m3_t(float f) {
  const u32 b = wg_q8m_bits(f);
  const float ax = __builtin_fabsf(f);
  const u32 cb = wg_q8m_magic(ax);
  u32 c = (wg_q8m_bits(ax + wg_q8m_flt(cb)) - cb) + (cb >> 20) - 1132u;
  c = c < 0x7eu ? c : 0x7eu;
  if (NANFIX) c = (f != f) ? 0x7fu : c;
  return c | ((b >> 24) & 0x80u);
}
WG_FN u32 wg_f32_to_e4m3(float f) { return wg_f32_to_e4m3_t<true>(f); }        // same bytes as the old function
WG_FN u32 wg_f32_to_e4m3_inner(float f) { return wg_f32_to_e4m3_t<false>(f); }  // NaN -> sign|0x7e (as wg_pack_q8_inner)

template <bool NANFIX>
WG_FN v2i wg_pack_q8_t(v8f d) {
  u32 w[2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
    w[h] = wg_f32_to_e4m3_t<NANFIX>(d[4 * h]) | (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 1]) << 8) |
           (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 2]) << 16) | (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 3]) << 24);
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}
WG_FN v2i wg_pack_q8(v8f d) { return wg_pack_q8_t<true>(d); }
WG_FN v2i wg_pack_q8_inner(v8f d) { return wg_pack_q8_t<false>(d); }

// ---- output (b): the E4M3-rounded value as F16 bits (every E4M3 value is exact in F16) ---------------------------
// CL: saturate at +-448 (inf too); NANFIX: NaN -> F16 NaN. Without CL the input must satisfy |x| <= 464 and not be
// NaN. v_cvt_pk_rtz_f16_f32 is exact here because the value has at most 4 significant bits.
template <bool NANFIX, bool CL>
WG_FN float wg_q8m_round(float f) {
  const float ax = __builtin_fabsf(f);
  const float C = wg_q8m_flt(wg_q8m_magic(ax));
  float r = (ax + C) - C;
  if (CL) r = wg_q8m_clamp(r, 0.0f, 448.0f);
  u32 rb = wg_q8m_bits(r);
  if (NANFIX) rb = (f != f) ? 0x7fc00000u : rb;
  return wg_q8m_flt(rb | (wg_q8m_bits(f) & 0x80000000u));  // sign last: a zero result keeps the input's sign
}
template <bool NANFIX, bool CL>
WG_FN v8h wg_q8_to_f16_t(v8f d) {
  u32 w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    w[i] = __builtin_bit_cast(u32, __builtin_amdgcn_cvt_pkrtz(wg_q8m_round<NANFIX, CL>(d[2 * i]),
                                                              wg_q8m_round<NANFIX, CL>(d[2 * i + 1])));
  return __builtin_bit_cast(v8h, w);
}
// F16 bits of q8(x): exact incl. NaN (F16 NaN) and the sign of a zero result.
WG_FN v8h wg_q8_to_f16(v8f d) { return wg_q8_to_f16_t<true, true>(d); }
// NaN -> +-0 (it cannot hide: see wg_pack_q8_inner's note); everything else exact.
WG_FN v8h wg_q8_to_f16_inner(v8f d) { return wg_q8_to_f16_t<false, true>(d); }
// For inputs proven within +-464 and not NaN (cf. wg_pack_q8_nc): no clamp, no NaN select. Bit-exact on that domain.
WG_FN v8h wg_q8_to_f16_nc(v8f d) { return wg_q8_to_f16_t<false, false>(d); }
