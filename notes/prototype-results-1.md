# Prototype results 1: FP8 decode and prefill kernels on the RX 7900 XT

Date: 2026-11-15. Run `build/runs/20261115-123231/` (gitignored logs), made by `bench/run-gpu.sh` (all phases).
Prototypes: `proto/gemv/fp8_gemv.hip` (decode), `proto/gemm/fp8_gemm.hip` (prefill), at commit a74ecec plus the
`time` mode. Baselines: llama.cpp 6d3568e31 `test-backend-ops perf` with `bench/llama-cpp-out-of-cache-shape.patch`.
Weight 16384 x 14336 (235 MB as FP8, above the 80 MB Infinity Cache). The card was shared with the WrongGPU
project's runs; every timed command held the shared GPU lock, but untimed work from other processes could still run.
Clock samples before runs: 1368-2447 MHz.

## Correctness (82 runs, all passed)

- Decode, M = 4093 (ragged), K = 4096, every variant, R = 1/2/4, n = 1..8:
  - variant A (F32 activations): NMSE about 4.5e-14 against a float64 reference, max |err| / sum|w*a| about 5e-8;
  - variant B (FP16 activations): NMSE about 4.2e-8, max about 4.3e-5 (the plan's FP16-rounding bound is about 8e-8);
  - nan mode (0x7F in row 5, 0xFF in the last row): exactly those rows NaN in every column, nothing else NaN;
  - ovf mode (activation 1e5): variant B reports the fallback and matches variant A to the printed digits.
- Prefill, M = K = 4096, n = 64, 100, 512 and nan mode: sampled NMSE about 1.2e-10 against a reference with
  FP16-rounded activations; NaN rows exact. This confirms the WMMA output layout used in the store.
- Full shape (prefill timed runs also check): NMSE about 1.6e-9.

## Decode (done criterion: within 3 % of Q8_0 at n = 1..8; spread rule where Q8_0 spread > 3 %)

Q8_0: median of 13 runs (10 baseline + 3 interleaved); spread = (max - min) / median. Ours: median over 3 rounds of
each run's median of 200 launches. Best rows-per-wave R per cell. Table from `proto/parse_runs.py`.

| n | Q8_0 us | Q8_0 spread | A us (R) | A / Q8_0 | B us (R) | B / Q8_0 |
|---|---|---|---|---|---|---|
| 1 | 353.2 | 7.1 % | 316.9 (1) | 0.897 | 325.8 (2) | 0.922 |
| 2 | 372.3 | 3.8 % | 321.3 (2) | 0.863 | 328.2 (2) | 0.881 |
| 3 | 380.4 | 2.1 % | 339.4 (2) | 0.892 | 334.0 (1) | 0.878 |
| 4 | 385.1 | 3.4 % | 349.0 (4) | 0.906 | 337.3 (2) | 0.876 |
| 5 | 388.8 | 2.8 % | 364.0 (4) | 0.936 | 337.0 (2) | 0.867 |
| 6 | 414.4 | 2.9 % | 377.5 (4) | 0.911 | 347.6 (1) | 0.839 |
| 7 | 446.7 | 2.1 % | 450.3 (4) | 1.008 | 354.7 (4) | 0.794 |
| 8 | 484.1 | 2.1 % | 479.6 (2) | 0.991 | 359.9 (4) | 0.743 |

- Every cell passes the gate. llama.cpp's own FP8 decode is 1096-1683 us here (3.1-3.6x Q8_0).
- Explained: A at n = 1 reads 741 GB/s of weights, 98 % of the 754 GB/s read ceiling (`bench/bandwidth.hip`) and the
  same as llama.cpp F16 (739 GB/s), so it is memory-bound as the instruction count predicted (3.7 VALU/weight).
  A slows from n = 7 (10.1-11.1 VALU/weight); B, at 6.0 VALU/weight for n = 8, stays near 650 GB/s.
- Not yet comparable to the last percent: our harness times each call with hipEvents; test-backend-ops times its
  own way, and Q8_0's time includes its activation quantization (as B's includes its conversion pass). The plan's
  harness cross-check (an F16 GEMV in our harness against llama.cpp's F16) is not done. A's margins at n = 7 and 8
  (about 1 %) are inside what that difference could move; B's 8-26 % are not. The decisive check is the same kernel
  inside llama.cpp, timed by test-backend-ops.
- Our run-to-run spread reached 5.8-7.0 % for some A R=4 and B R=4 cells (card shared).

## Prefill (second goal): not met

| n | ours us (TFLOPS) | llama.cpp F8 us | Q8_0 us | F16 us |
|---|---|---|---|---|
| 64 | 1683 (17.9) | 2118 | 689 | 939 |
| 128 | 1765 (34.1) | 2822 | 1261 | 1590 |
| 256 | 3302 (36.4) | 3241 | 2306 | 2251 |
| 512 | 6188 (38.9) | 5222 | 4250 | 4293 |
| 1024 | 11573 (41.6) | 9610 | 8357 | 8737 |
| 2048 | 23030 (41.8) | 18103 | 16785 | 17195 |
| 4096 | 46466 (41.4) | 35635 | 34566 | 35287 |

- llama.cpp rows are single runs; ours are medians of 20 launches with a heavy tail (n = 512: p90 18.7 ms).
- Ours beats llama.cpp's FP8 only at n = 64 and 128 and reaches about 72 % of Q8_0 at large n. Known gaps in this
  first cut: no prefetch or double buffering (global-load latency is exposed between two barriers per K step), and
  n is padded to 128 (n = 64 wastes half the tile).

## Next

1. Harness cross-check for decode, then the same kernels inside llama.cpp's `mmvf` path timed by
   `test-backend-ops` (the real build; Feature playbook).
2. Prefill: prefetch the next K tile into registers during the WMMA loop, a 64-wide N tile for small n; re-time.
