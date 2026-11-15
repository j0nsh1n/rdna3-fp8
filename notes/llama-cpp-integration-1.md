# llama.cpp integration 1: shift-widening FP8 decode inside `mmvf`

Date: 2026-11-15. Patch: `bench/llama-cpp-fp8-mmvf-hip.patch` (on llama.cpp 6d3568e31, `ggml/src/ggml-cuda/mmvf.cu`
only; apply with `git -C build/llama.cpp apply`). Run: `build/runs/20261115-130234/` (`bench/run-gpu.sh llama`).
Not committed inside llama.cpp: their `AGENTS.md` forbids agent commits, pushes and PRs; any upstream offer is the
owner's own work and decision.

## What the patch does

In `mul_mat_vec_f` for `T = ggml_fp8_e4m3_t`, HIP builds only: when the row length is a multiple of 16 and `x`,
`y` (and the fused `gate_x`) are 16-byte aligned, each thread loads 16 weight bytes, widens them with the shift
(value * 2^-8) and accumulates `fmaf(half -> float, y, sum)` against the F32 activations (variant A of the
prototype). After the loop each thread scales its partial sums by 256, or sets them to NaN if it saw a NaN code; the
existing block reduction carries the NaN to the output. The gate row has its own NaN flag, because
`swiglu_oai` clamps the gate with fminf/fmaxf and would turn a NaN into a number. Every other case runs the
original loop.

## Correctness

- `test-backend-ops test -b ROCm0 -p f8_e4m3`: 450/450 OK (157 MUL_MAT, 228 MUL_MAT_FUSION, plus the other F8 ops).
- Mutations: scale 255 instead of 256 passes all 305 MUL_MAT/MUL_MAT_FUSION F8 cases (llama.cpp's NMSE gate is 5e-4;
  a 1/256 error is about 1.5e-5), so their tests cannot see small scale errors. Scale 128: 244 FAIL, 61 OK, so the
  new path is exercised by 244 cases; the other 61 are unaligned or larger-n cases on other paths.
- NaN codes are not in llama.cpp's test data; the NaN logic is covered by the prototype tests only.
- Generated code (mmvf.cu compiled alone with llama.cpp's flags): no spills for any batch size; up to 108 VGPRs.

## Timing (`test-backend-ops perf`, 16384 x 14336, 3 rounds, median us)

| n | Q8_0 | F8 patched | / Q8_0 | F8 unpatched (run 20261115-123231) | F16 |
|---|---|---|---|---|---|
| 1 | 357.0 | 339.1 | 0.95 | 1095.8 | 683.1 |
| 2 | 379.6 | 354.4 | 0.93 | 1144.0 | 695.3 |
| 3 | 386.5 | 394.9 | 1.02 | 1200.8 | 714.9 |
| 4 | 390.7 | 557.4 | 1.43 | 1236.6 | 773.8 |
| 5 | 392.4 | 887.8 | 2.26 | 1303.7 | 772.2 |
| 6 | 415.9 | 927.0 | 2.23 | 1317.9 | 784.1 |
| 7 | 448.4 | 1164.7 | 2.60 | 1611.9 | 771.8 |
| 8 | 484.9 | 1286.8 | 2.65 | 1683.0 | 782.5 |

- Same harness as Q8_0, so n = 1..3 is settled: the exact FP8 path matches Q8_0 (0.93-1.02x).
- Faster than unpatched FP8 at every n (3.2x at n = 1, 1.3x at n = 8), but far from the prototype at n >= 4
  (prototype variant A: 349 us at n = 4, 480 us at n = 8).
- Explained by structure, not spills: `mmvf` runs one weight row per block with up to 256 threads, so every row
  reloads all n activation vectors (n x 64 bytes of F32 activations per 16 weight bytes). The prototype got its n >= 4
  speed from 2-4 rows per wave sharing those loads (R = 1 in the prototype was also 1.3-1.4x Q8_0 at n >= 4).
  The jumps at n = 5 and 7 are not explained yet.

## Step 2: rows per warp for n = 2..8 (same patch file, updated)

`mul_mat_vec_fp8_rows<ncols_dst, rows_per_warp>`: a separate FP8 kernel in `mmvf.cu` (HIP only), used for
ncols_dst 2..8 without fusion or ids when shapes and pointers are 16-byte aligned (channels and samples supported);
everything else keeps `mul_mat_vec_f`. Each warp does 2 or 4 rows (prototype's best: 2 for n = 2, 3, 8; 4 for
n = 4..7) and loads each activation chunk once for all its rows. NaN codes go through the warp sum as NaN.

- `test-backend-ops test -b ROCm0 -p f8_e4m3`: 450/450 OK. Mutation (scale 128 in the new kernel): 19 MUL_MAT cases
  FAIL, covering every n from 2 to 8, so the kernel is exercised. Restored and rebuilt.
- Generated code: no spills (53-112 VGPRs); loop VALU per weight 4.9 (n = 2) to 11.3 (n = 8), the same as the
  prototype's variant A.
- Timing, run `build/runs/20261115-131702/`, `test-backend-ops perf`, 3 rounds, median us:

| n | Q8_0 | F8 rows | / Q8_0 | F8 step 1 | F16 |
|---|---|---|---|---|---|
| 1 | 366.3 | 321.7 | 0.88 | 339.1 | 679.2 |
| 2 | 377.8 | 328.3 | 0.87 | 354.4 | 673.4 |
| 3 | 378.1 | 346.9 | 0.92 | 394.9 | 714.1 |
| 4 | 382.4 | 365.5 | 0.96 | 557.4 | 760.0 |
| 5 | 395.4 | 372.7 | 0.94 | 887.8 | 765.6 |
| 6 | 421.2 | 393.6 | 0.93 | 927.0 | 770.6 |
| 7 | 448.1 | 505.6 | 1.13 | 1164.7 | 769.3 |
| 8 | 482.8 | 515.3 | 1.07 | 1286.8 | 780.9 |

(n = 1 uses the step 1 path; its change between runs is run-to-run variation, Q8_0's n = 1 spread is 8.3 %.)

- n = 1..6 beat Q8_0 in llama.cpp's own harness. n = 7 and 8 miss the 3 % gate by 13 % and 7 %.
- Explained: the code is the prototype's, so the prototype's parity at n = 7, 8 (450, 480 us) came from its own
  timing harness, as flagged in prototype-results-1. The PyTorch harness also gives about 500 us there
  (notes/pytorch-ext-1.md). At n >= 7 variant A is bound by its multiply instructions: one v_fma_mix_f32 per weight
  per column plus about 2.5 per weight for widening and the NaN check (about 10-11 VALU per weight).

## Step 3: non-temporal weight loads; n = 7, 8 closed

The step 2 kernel and the prototype's variant A compile to nearly the same loop (n = 8, 2 rows: 548 vs 539
instructions, 360 vs 356 VALU, the same 34 global_load_b128), yet llama.cpp's was 8-13 % slower at n = 7, 8. The one
difference: the prototype loads weights with `__builtin_nontemporal_load` (`global_load_b128 ... slc dlc`), the
llama.cpp kernel without. Weights are read once while each activation chunk is reused by every row; streaming the
weights past the cache keeps the activations in it. At 11 VALU per weight the kernel used only about 40 % of the
measured instruction rate, so it was waiting on memory, not instruction-bound as stated in step 2.

Experiments in llama.cpp's harness (temporary env override, removed; `build/runs/sweep-*.log`, 2 rounds each):

| setting | n = 7 / Q8_0, rows 1 / 2 / 4 | n = 8 / Q8_0, rows 1 / 2 / 4 |
|---|---|---|
| step 2 | 1.42 / 1.04 / 1.10 | 1.46 / 1.10 / 1.18 |
| + `__launch_bounds__(128)` | 1.43 / 1.07 / 1.10 | 1.47 / 1.12 / 1.18 (no effect, not kept) |
| + non-temporal weight loads | 1.36 / 0.93 / 1.05 | 1.39 / 0.93 / 1.12 |

With non-temporal loads the best rows per warp are 2 for n = 2, 7, 8 and 4 for n = 3..6 (n = 3: 0.875 vs 0.881,
kept at 2). llama.cpp has no non-temporal loads elsewhere in `ggml-cuda` (grep), so this is a new, if small,
pattern for them.

Final patch (`bench/llama-cpp-fp8-mmvf-hip.patch`, 200 lines, ASCII), run `build/runs/20261115-134919/`:
`test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_FUSION -p f8_e4m3` passes; `perf`, 3 rounds, median us:

| n | Q8_0 | F8 final | / Q8_0 |
|---|---|---|---|
| 1 | 337.9 | 312.3 | 0.92 |
| 2 | 374.5 | 314.8 | 0.84 |
| 3 | 383.9 | 335.3 | 0.87 |
| 4 | 382.4 | 342.2 | 0.89 |
| 5 | 386.1 | 341.8 | 0.89 |
| 6 | 409.1 | 348.9 | 0.85 |
| 7 | 442.1 | 414.1 | 0.94 |
| 8 | 479.2 | 447.4 | 0.93 |

Done criterion (decode within 3 % of Q8_0 at n = 1..8) is met in llama.cpp's own harness, with exact FP32
activations and llama.cpp's NaN semantics. Spreads this run: 0.2-2.9 % (3 rounds).

## Options for n >= 7 (superseded by step 3)

1. Fewer instructions per product with exact results: tried as prototype variant C (widened weight converted to
   FP32 with v_cvt_f32_f16, exact, then plain FP32 FMAs that may dual-issue). The compiler pairs only a small share
   into v_dual_fmac_f32 (VOPD needs operands in different register banks; HIP gives no control), so loop VALU per
   weight barely moves (n = 8: 11.7 -> 10.4 at R = 1, none at R = 2, 4). Timed in the prototype harness, 2 rounds
   (`build/runs/variant-c/ac-n78.log`): best C vs best A is 432 vs 441 us at n = 7 and 464 vs 478 us at n = 8
   (2-3 %, within noise); C at R = 1 is 1.6-1.9x slower. Results identical to A (NMSE 4.5e-14). Not pursued.
2. FP16 activations (variant B) for n >= 7 only: 5-6 VALU per weight; changes results slightly relative to
   llama.cpp's FP8 path (the owner allowed B only as a labelled faster option).
3. Accept: exact path wins at n = 1..6, loses 7-13 % at n = 7, 8 (still 2.3-2.5x faster than unpatched FP8).
