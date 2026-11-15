# Exact FP32 -> E4M3 on RDNA 3 (gfx1100): faster software conversion

Card: Radeon RX 7900 XT (gfx1100), hipcc 7.1 (LLVM 20), wave32. Labels: MEASURED = observed here; INFERRED = reasoning
from those observations. Standalone prototypes only; nothing in any repository was touched. Everything below is
reproduced by `./run.sh` (memory cap 6 GB, no swap, nice, shared GPU lock built in).

## Result in one paragraph

An exact conversion exists at about 0.4x the baseline's cost. For output (b) (F16 bits ready as the next matrix
operand) `H1` costs **7.7 cycles per value per wave** against 17.9 for the baseline quantizer alone (and 20.3 for
baseline quantizer plus the repo's widening): 2.3x and 2.6x faster. It is bit-exact on all 2^32 inputs, NaN included. For output (a)
(code bytes) `A6` costs **11.3** against 17.9 (1.6x), also bit-exact on all 2^32 inputs. (MEASURED.) The idea that
worked is the one in the task: round in the FP32 domain, then one packed `v_cvt_pk_rtz_f16_f32` per two values. The
rounding is a single FP32 add-and-subtract of a magic constant derived from the value's own exponent.

## 1. Semantics (what "exact" means here)

Read from the repo (`wg_platform.h` `wg_f32_to_e4m3`, `reference/fp8.py` `q8`) and confirmed by the reference below:

| Input | Result byte |
|---|---|
| finite, rounded to the E4M3 grid, nearest, ties to even; the E4M3 subnormal step is 2^-9 below 2^-6 | sign\|code |
| \|x\| >= 448 (rounds to or beyond the maximum) and +-inf | sign\|0x7e (+-448) |
| NaN (any payload) | sign\|0x7f (sign of the NaN kept) |
| FP32 denormal inputs, +-0, anything below 2^-10 | sign\|0x00: a signed zero (the sign is kept) |

FP32 denormals: `wg_f32_to_e4m3` computes `rint(min(a, 2^-6) * 512)` on them, which is 0, so they give a signed zero; the
q8 definition gives the same. All variants below produce the same on the GPU with default denormal mode (the
"fp32-denormal" class has 0 mismatches for every exact variant). (MEASURED.)

Where the HLSL `wg_e4m3_hw` differs: it models the hardware encoder, not the saturating form. For \|x\| <= 464 its bytes
equal the saturating form's; above 464 it gives sign\|0x7f (the NaN code) and for NaN 0xff. The task text says
"saturating to +-448", which is the saturating form (`wg_f32_to_e4m3`, `q8`, and `wg_pack_q8` through `wg_sat448`), so
that is what is matched and measured. The `hw` form was not built; INFERRED: it would add two compares per value.

Output (b) is the F16 bits of q8(x): exact values, sign of a zero kept, \|x\|>=448 -> +-448 = 0x5f00/0xdf00. A NaN input
gives any F16 NaN. (The repo's current widening turns the NaN code 0x7f into +-480, a finite value; (b) is stricter.)

## 2. Reference (MEASURED)

`ref.h`/`refcheck.c`: `def_code` is written from the definition in double (`frexp`, spacing, `rint`, then the code by
exact search in the E4M3 value table). `fast_code` is the repo's bit form. Results (`refcheck_result.txt`):

- `fast_code` == `def_code` for **all 2^32** inputs (4 threads): 0 mismatches.
- `def_code` vs Python `fp8.py` (`fp8_encode`, `q8`) on a 4,902,746-value sample: 0 mismatches. The sample has the midpoint
  of every pair of adjacent E4M3 values with +-2 FP32 neighbours (both signs), every E4M3 value with neighbours, 448/464/480
  and neighbours, +-0, +-inf, 130+ NaN payloads (both signs), 200,000 denormal and tiny patterns each sign, 3M uniform bit
  patterns, 1.5M values in 2^-17..2^12. The value check against `q8` also passes (0 mismatches).

## 3. Variants

All work on 4 values: a byte variant returns one dword of 4 codes (as `wg_pack_q8` packs), a half variant returns 2
dwords of F16 pairs. Source: `variants.h` (every variant), `vlist.h` (the table).

The rounding core (variant "MA"): z = |x| clamped to [2^-6, 448] (one `v_med3_f32`), C = 1.5 * 2^(E+20) with E the exponent
of z (`v_and` + `v_add` on the bits), then `y = |x| + C`, `r = y - C`. The FP32 add rounds |x| to a multiple of
2^(E-3), nearest even, because C's mantissa is even; the subtract is exact. The floor 2^-6 gives the subnormal step 2^-9.
Then clamp r to 448 (`v_med3`), merge the sign (`v_and_or`, keeps the sign of a zero), NaN select, and
`v_cvt_pk_rtz_f16_f32` for two values. `v_med3_f32` is used for clamps because LLVM puts a canonicalising `v_max x,x`
in front of every `fminf`/`fmaxf` (gfx11, IEEE mode), which costs one instruction each.

| Name | Out | What | Domain / NaN rule |
|---|---|---|---|
| A0 | (a) | baseline: faithful port of `wg_f32_to_e4m3`, 4 packed | exact |
| H0 | (b) | baseline code, then the repo's widening (bit form + `v_pk_mul_f16` by 256) | NaN -> +-480 (finite) |
| A1 | (a) | MA, code from the rounded float: `|r| * 2^-120` has the code in bits 26:20 (E4M3 subnormals are FP32 denormals after the scaling), `>> 20`, min 0x7f | exact |
| A3 | (a) | integer add-and-shift for normals + magic add for subnormals, merged by one min/max | exact |
| A6 | (a) | MY: code straight from the magic sum: code = (bits(C)>>20) - 1132 + (bits(y) - bits(C)), min 0x7e | exact |
| A7 | (a) | A6 without the NaN select | NaN -> sign\|0x7e |
| A5 | (a) | MA bounded: no clamp, no NaN select | needs \|x\| <= 464, not NaN |
| H1 | (b) | MA + NaN select + sign merge + packed cvt | **exact** (NaN -> F16 NaN) |
| H4 | (b) | integer add-and-mask for normals + magic add for subnormals + select | exact |
| H5 | (b) | H1 without the NaN select | NaN -> +-0 |
| H6 | (b) | H1 without clamp and NaN select | needs \|x\| <= 464, not NaN |
| H9 | (b) | MA signed: sign falls out of the arithmetic, no sign merge | value-exact; a zero result from a negative input is +0 |
| H10 | (b) | H9 without the NaN select | as H9, NaN -> garbage (+-0 or NaN) |
| H11 | (b) | H9 bounded | needs \|x\| <= 464, not NaN; zero sign as H9 |
| A2/H2 | | "MB": C = max(\|x\|*2^20, 2^14) (no exponent mask) | **rejected**, see below |
| H3 | (b) | "MC": Veltkamp split (x*(2^20+1)) + magic for subnormals | **rejected** |

## 4. Exhaustive check, all 2^32 FP32 bit patterns (MEASURED; `exh_all.txt`)

Each variant ran over every bit pattern in 64 chunks of 2^26 on the GPU; the host compared with the CPU reference
(4 threads). Classes of the input: zero, FP32-denormal, E4M3-subnormal range (< 2^-6), normal range (< 448), 448..inf finite, inf,
NaN. Output (b) is compared as F16 bits (NaN: any F16 NaN). "Total" counts all mismatches.

| Variant | Total | zero | fp32-denormal | < 2^-6 | normal | 448..inf | inf | NaN | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A0 baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| **A1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| **A3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| **A6** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| A7 | 16,777,214 | 0 | 0 | 0 | 0 | 0 | 0 | 16,777,214 | exact except NaN (-> 0x7e by design) |
| A5 | 1,999,634,432 | 0 | 0 | 0 | 0 | 1,999,634,430 | 2 | 0 | exact for \|x\|<=464 (0 mismatches there) |
| A2 (MB) | 224 | 0 | 0 | 0 | 224 | 0 | 0 | 0 | **rejected** |
| H0 baseline + widen | 16,777,214 | 0 | 0 | 0 | 0 | 0 | 0 | 16,777,214 | NaN -> +-480 as today; all else exact |
| **H1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| **H4** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | exact |
| H5 | 16,777,214 | 0 | 0 | 0 | 0 | 0 | 0 | 16,777,214 | exact except NaN (by design) |
| H6 | 1,999,634,432 | 0 | 0 | 0 | 0 | 1,999,634,430 | 2 | 0 | exact for \|x\|<=464 (0 mismatches there) |
| H9 | 981,467,137 | 1 | 8,388,607 | 973,078,529 | 0 | 0 | 0 | 0 | **all are the sign of a zero** (counter "wrong-sign zero" = the total); values exact |
| H10 | 998,244,351 | 1 | 8,388,607 | 973,078,529 | 0 | 0 | 0 | 16,777,214 | zero sign + NaN, nothing else |
| H11 | 2,981,101,569 | 1 | 8,388,607 | 973,078,529 | 0 | 1,999,634,430 | 2 | 0 | zero sign + above 464; exact values for \|x\|<=464 |
| H2 (MB) | 224 | 0 | 0 | 0 | 224 | 0 | 0 | 0 | **rejected** (first: 0x3cfffff1) |
| H3 (MC) | 245,530,624 | 0 | 0 | 0 | 245,530,624 | 0 | 0 | 0 | **rejected** |

Reading:
- No exact variant has any mismatch in any class, FP32 denormals and NaNs included. The "outside denormals" rule is met
  with room to spare; denormal behaviour is the baseline's (signed zero). The count "1" in class zero for H9/H10/H11 is -0.
- MB fails in 224 cases where `|x| + C` crosses into the next binade (|x| mantissa all ones in the top 20 bits): C is
  not a multiple of the grid there. MC (Veltkamp) does not break ties to even. Both are rejected, not tuned.
- **The check can fail (MEASURED, `flip_result.txt`):** with the reference's tie rule changed (ties to odd in the normal
  range, ties up in the subnormal range) the baseline A0, A1 and H1 each report 246 mismatches (236 normal, 10
  subnormal-range). Only exact-tie inputs differ, which is why the count is small; it is the same for all three because
  they all implement ties-to-even.
- Drop-in (`wg_e4m3_rdna3.h`, section 7) re-checked through `v8f`: `wg_pack_q8` and `wg_pack_q8_inner`, `wg_q8_to_f16`,
  `wg_q8_to_f16_inner`: 0 mismatches against their stated semantics over 2^32 inputs; `_nc` has 0 mismatches for
  \|x\|<=464 (`dropin_result.txt`).
- `v_cvt_pk_rtz_f16_f32` (round toward zero) is exact here: 0 mismatches. The single-value RNE `v_cvt_f16_f32` was
  not separately exercised; INFERRED exact for the same reason (values have at most 4 significant bits).

## 5. Timing (MEASURED; `timing_table.txt`, `timing_raw.txt`)

Method of `RESULTS-VALU.md`: 672 workgroups of 256 (5376 waves of 32 = 32 per SIMD; occupancy 16 for every kernel), 16
values per lane per iteration, 3 warm-ups + 10 timed reps at N and 2N iterations, `fixed = 2 med t(N) - med t(2N)`,
cost = (t(2N) - fixed)/2N. Cycles come from a CALIB kernel (128 independent `v_fma_f32` per iteration, 1 per cycle); the clock
drifts on this machine (CALIB gave 2.3 to 4+ GHz-equivalents across hours, 8-17 % spread within a run), so **every
variant was timed directly after its own CALIB run, 5 such pairs, and the table is the median of the 5 ratios**. LLVM
dual-issue (VOPD) is switched off (`-mllvm -amdgpu-enable-vopd=0`; 0 `v_dual_*` in the listing) so that one VALU
instruction takes one slot, as with ACO; with it on, CALIB itself ran at 2 FMAs per cycle. The harness costs one
`v_add_nc_u32` per value plus the xor fold; BASE_A/BASE_H are the same harness with no conversion and are subtracted
(1.56 and 2.01 cycles/value gross).

| Variant | Output | Cycles per 16 values per wave, median [min-max of 5] | **Cycles / value (net of harness)** | VALU instr. / value (straight-line ISA for A0, A6, H1, H9, H11; loop-based incl. ~1.5-2 harness otherwise) | vs baseline |
|---|---|---|---|---|---|
| A0 baseline | (a) | 310.8 [235-336] | **17.86** | 19.5 | 1.00x |
| A1 | (a) | 239.9 [226-257] | 13.43 | 16.3 (loop: 260 instr. / 16 values, harness included) | 1.33x |
| A3 | (a) | 218.1 [208-235] | 12.06 | 14.1 (loop 226/16, harness included) | 1.48x |
| **A6** | (a) | 206.3 [192-231] | **11.33** | 13.25 | **1.58x** |
| A7 (NaN -> 0x7e) | (a) | 183.6 [173-198] | 9.91 | 12.2 (loop 196/16, harness included) | 1.80x |
| A5 (bounded) | (a) | 205.8 [194-213] | 11.30 | (ISA 207/16 incl. harness) | 1.58x |
| H0 baseline + widen | (b) | 357.3 [346-369] | 20.32 | 23.5 (loop 376/16, harness included) | 1.00x (b) |
| **H1** | (b) | 155.5 [139-170] | **7.71** | 9.5 | **2.64x** (b), 2.3x vs A0 |
| H4 | (b) | 202.7 [194-217] | 10.66 | (integer path + select) | 1.9x |
| H9 (zero sign may differ) | (b) | 146.6 [125-158] | 7.16 | 8.5 | 2.84x |
| H5 (NaN -> 0) | (b) | 140.7 [132-144] | 6.79 | 7.5 (loop 144/16 = 9 incl. harness) | 3.0x |
| H6 (bounded) | (b) | 123.4 [112-128] | 5.71 | 6.5 (loop: 128/16 = 8 incl. harness) | 3.6x |
| H10 | (b) | 119.2 [106-127] | 5.44 | | 3.7x |
| H11 (bounded, zero sign may differ) | (b) | 99.9 [95-113] | 4.24 | 5.5 | 4.8x |

Notes:
- Scaling: t(2N)/t(N) median 1.966 (range 1.64-2.09) and t(1344 WG)/t(672 WG) median 1.938 (1.64-2.14) over 80 runs. A perfectly
  linear kernel gives 2.0; the shortfall is the fixed launch cost (as in `RESULTS-VALU.md`). The few outliers are
  desktop activity: reported ranges are wide for A0 (235-336) and H1 (139-170), so single digits in the table are +-10 %.
- The baseline here is 17.9 cycles/value against 16.6 measured with Vulkan/ACO in `RESULTS-VALU.md` (the ISA differs:
  LLVM emits 19.5 VALU instructions/value, ACO ~16). Ratios, not absolute cycles, carry over; the saving below is stated
  both ways.
- H0 (baseline + widening) = 20.3 means the repo's widening costs about 2.4 cycles/value after LLVM, less than the
  4.4 of the Vulkan bit-form kernel.
- Not run: the byte variants were not exercised through memory stores; the timed kernel folds the dword with xor.

## 6. The best variant's ISA

Best exact (b), `H1` (`isa.hip`, `isa.s`; four values -> two F16 pairs, 38 VALU instructions = 9.5 per value, two more
are address shifts of the test harness). The pattern per value is
`v_med3_f32` (clamp z) - `v_and_b32` (exponent) - `v_add_nc_u32` (C) - `v_add_f32` (|x| + C) - `v_sub_f32` (- C) -
`v_med3_f32` (clamp to 448) - `v_cmp_o_f32` + `v_cndmask_b32` (NaN) - `v_and_or_b32` (sign), then one
`v_cvt_pk_rtz_f16_f32` per two values:

```
v_med3_f32 v5, |v1|, s0, 0x43e00000      ; z = clamp(|x|, 2^-6, 448); s0 = 2^-6          (x4)
v_and_b32_e32 v5, 0x7f800000, v5         ; exponent of z                                  (x4)
v_add_nc_u32_e32 v5, 0xa400000, v5       ; C = 1.5 * 2^(E+20)                             (x4)
v_add_f32_e64 v9, |v1|, v5               ; y = |x| + C: rounds |x| to the grid, ties even (x4)
v_sub_f32_e32 v5, v9, v5                 ; r = y - C, exact                               (x4)
v_med3_f32 v5, v5, 0, 0x43e00000         ; saturate at 448 (inf too)                      (x4)
v_cmp_o_f32_e32 vcc_lo, v1, v1           ; NaN test                                       (x4)
v_cndmask_b32_e32 v5, 0x7fc00000, v5, vcc_lo ; NaN -> qNaN                                (x4)
v_and_or_b32 v1, 0x80000000, v1, v5      ; sign (a zero result keeps its sign)            (x4)
v_cvt_pk_rtz_f16_f32_e32 v1, v1, v2      ; two F16 values, exact                          (x2)
```
Per value: 7 vector ops for the rounding and clamp, 2 for NaN, 1 for the sign merge (the `x2` cvt is 0.5), 9.5 in total. Without
NaN (H5): 7.5, with no clamp too (H6): 6.5, signed (H9 without NaN: H10): 6.0/5.5.
The full loop listing (all 16 values, harness included) is in `asm/` after `./run.sh isa`; `asm_counts.txt` has its
instruction mix (H1: 176 instructions in the loop body for 16 values incl. 32 harness adds and 4 xor3).
Best exact (a), `A6`: 13.25 VALU per value: the same C and y, then `v_sub_nc_u32` (k), `v_lshrrev_b32` (bits(C)>>20),
`v_add3_u32` (code), `v_min_u32` (0x7e), NaN `v_cmp_o`+`v_cndmask`, sign `v_and_or`, and the pack.

## 7. Per-layer-step saving for the FP16 route (INFERRED, built on the MEASURED numbers)

Step budget from `RESULTS-VALU.md` section 6 (route 1): widen A and B codes 140.8 + F16 WMMA 33.1 + quantize 8 outputs
133.0 + half-wave exchange 14.7 = 321.6 cycles, one 16x16x16 tile, 8 output values per lane.

What output (b) removes from the widening: the layer's output is produced directly as the F16 operand, so the activations
need no E4M3 -> F16 widening in the next layer's step. Of the 141 widening cycles, the activation operand's share is
70.4 (16 codes per lane, 2 x 35.2); the weights' share (70.4) remains, because weights stay stored as E4M3 codes. The
quantize step is replaced by the (b) rounding, which writes F16 without a code byte in between.

| Route | Quantize (8 values) | Widen | Step total | Saving |
|---|---|---|---|---|
| Today (Vulkan numbers) | 133.0 | 140.8 | 321.6 | - |
| (a) A6 as the quantizer, widen unchanged (ratio 11.33/17.86 x 133.0) | 84.4 | 140.8 | 273.0 | 48.6 (15 %) |
| **(b) H1: round to F16 directly, widen weights only** (ratio 7.71/17.86 x 133.0) | 57.4 | 70.4 | **175.6** | **146.0 (45 %)** |
| (b) H1, with the HIP-measured 8 x 7.71 instead of the scaled number | 61.7 | 70.4 | 179.9 | 141.7 (44 %) |
| (b) H6 (bounded input, 5.71) | 42.4 | 70.4 | 160.6 | 161 (50 %) |

The (b) row sums: 57.4 round + 70.4 weight widening + 33.1 WMMA + 14.7 exchange = 175.6; saving 321.6 - 175.6 = 146.0
cycles, about 45 %. If the exchange can be folded into the rounding (not examined), more. Stored E4M3 outputs (final layer, anything written to memory) still need
(a) once per output: A6 costs 90 cycles per 8 values (HIP) and is not on the per-step critical path of the F16 route.
Caveats: the ratios come from HIP/LLVM code; the budget numbers come from Vulkan/ACO, so the two sets are not the same
compiler, and the vector glue adds to the matrix time (section 5 of `RESULTS-VALU.md`), so the saving is a saving of
issue slots on a SIMD that is busy with the matrix anyway; real time saved needs a re-run of the full step.

## 8. Ideas tried and not tried

- Tried and exact: exponent-derived magic constant (MA: H1, A1, A6), integer add-and-mask + magic (MD: H4, A3 form). MA is
  faster because it has no select between a normal and a subnormal path.
- Tried and rejected: MB (scaled magic, no mask: 224 binade-crossing cases), MC (Veltkamp split: ties wrong in 245M
  inputs). Not worth fixing: a repair needs the mask and add MA already has.
- Value-exact variants that drop the sign merge (H9, H10, H11) save one op per value (the sign of a zero result is lost);
  harmless for a matrix operand's value, but not bit-exact: use only if -0 vs +0 is irrelevant downstream.
- Not built: lookup tables (an LDS table per value costs a load and the address math; RESULTS-VALU already found the LDS
  table slower than bit tricks for widening), `v_frexp`/`v_ldexp` (same 3 ops as and+add+med3 for the constant, no gain),
  hardware FP32 -> F16 followed by packed F16 rounding (double rounding: a value just above an E4M3 midpoint can land
  exactly on it in F16; making it exact needs a sticky bit, about the cost saved; INFERRED, not run), two values per
  packed instruction (the rounding needs FP32 mantissa bits below bit 16, so the 16-bit packed integer ops cannot hold
  it; INFERRED, not run). The layer-bound idea is the "bounded" variants: H6/A5 need \|x\| <= 464, no NaN, and save
  2 ops per value; H11 saves 3.

## 9. Drop-in

`wg_e4m3_rdna3.h`: same names and signatures as `wg_platform.h` (`wg_f32_to_e4m3`, `wg_pack_q8`, `wg_pack_q8_inner`)
plus new `wg_q8_to_f16` (exact), `wg_q8_to_f16_inner` (NaN -> +-0), `wg_q8_to_f16_nc` (\|x\|<=464, no NaN) returning `v8h`.
`wg_pack_q8_nc` is not redefined (not benchmarked as a separate form; A5 is its analogue). Needs `WG_FN`, `u32`, `v2i`,
`v8f`, `v8h` from `wg_platform.h`. Built and checked over all 2^32 inputs by `dropin_test.hip` (0 mismatches against stated
semantics), compiled with default denormal mode.

## 10. Files

Sources: `ref.h`, `refcheck.c`, `pycheck.py` (CPU reference and Python cross-check), `variants.h`, `vlist.h`, `exh.hip`
(exhaustive), `bench.hip`, `timing.sh`, `timing.py`, `isa.hip`, `asm.sh`, `loopbody.py`, `dropin_test.hip`,
`wg_e4m3_rdna3.h`, `run.sh`. Results: `refcheck_result.txt`, `exh_all.txt`, `flip_result.txt`, `dropin_result.txt`,
`timing_raw.txt`, `timing_table.txt`, `asm_counts.txt`, `asm_resources.txt`, `isa.s`. `probe/` holds the first NaN/med3 probes.
