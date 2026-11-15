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

## Options for n >= 4

1. Rows per block in the FP8 path (a template parameter or a separate FP8 kernel in `mmvf.cu`): the prototype shows
   Q8_0 parity is reachable. It is a structural change to a shared kernel, so llama.cpp's guidelines ask to pause and
   discuss before writing it.
2. FP16 activations (prototype variant B, 8-26 % faster than Q8_0): changes results relative to llama.cpp's FP8
   path; a bigger discussion upstream.
