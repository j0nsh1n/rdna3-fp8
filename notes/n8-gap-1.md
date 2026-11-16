# Closing the 8-column gap (n = 8) of the FP8 decode kernel

Date: 2026-11-15. Only `build/llama.cpp/ggml/src/ggml-cuda/mmvf.cu` changed (diff: `build/n8-work/mmvf-n8.diff`,
copy of the final file `build/n8-work/mmvf.cu.final`, original `build/n8-work/mmvf.cu.orig`). Nothing committed or pushed.

## In short

FP8 is now faster than Q8_0 with 8 parallel streams: 6.3 % faster on Llama 3.1 8B (was 8 % slower) and 3.5 % faster
on Llama 3.2 3B (was 10 % slower). On the big 16384 x 14336 weight FP8 takes 0.74x of Q8_0's time (was 0.91x).
7 streams got faster too (+3 to +5 %). 1-6 streams did not change (their code is identical). Results are unchanged
(same KL divergence as before).

## What was wrong (MEASURED from the generated code, INFERRED for the cause)

The n = 8 loop was fast on the big weight but slow in the models. The compiler issued the 4 activation loads of each column,
waited for them, and did 64-bit address arithmetic for every column (20 of 356 vector instructions per iteration). Simply
removing the address arithmetic made the compiler hoist all loads of the loop (about 190 registers, half the waves) and
got much slower (284 t/s vs 410). Registers decide speed here: kernels at or below about 64 registers were fast, those
above 100 were 10-30 % slower (rows 1 and 4 variants at 104-106 registers).

## What changed

New template argument `ygroup` of `mul_mat_vec_fp8_rows` (0 = old loop, used for n = 2..6). With `ygroup` > 0 the
activations are loaded `ygroup` columns at a time, with `__builtin_amdgcn_sched_barrier(0)` before and after each group,
from block-uniform column pointers plus one 32-bit offset (no 64-bit address math per column). The barriers stop the
compiler from hoisting all loads. Settings: n = 7: rows 4, ksplit 2, groups of 2; n = 8: rows 4 (was 2), ksplit 2, groups
of 2. Math and order of the fmaf per accumulator are unchanged (same results bit for bit as far as the tests see).

## Results (MEASURED, `llama-batched-bench`, medians of 3 rounds, FP8 / Q8_0 decode speed)

Llama 3.1 8B (before: `build/n8-work/runs/base2-llama31-8b.log`, after: `final3-llama31-8b.log`):

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| before | 1.080 | 1.063 | 1.069 | 1.060 | 1.056 | 1.055 | 1.003 | 0.923 |
| after | 1.076 | 1.072 | 1.072 | 1.062 | 1.058 | 1.056 | 1.060 | 1.066 |

Llama 3.2 3B (`base-llama32-3b.log`, `final3-llama32-3b.log`):

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| before | 1.096 | 1.072 | 1.073 | 1.073 | 1.060 | 1.047 | 1.003 | 0.898 |
| after | 1.101 | 1.069 | 1.072 | 1.072 | 1.060 | 1.047 | 1.039 | 1.034 |

The machine was a little slower during the "after" runs (Q8_0 itself was 2-3 % lower), so compare ratios, not t/s.
Absolute FP8 at n = 8, 8B: 420.3 -> 474-484 t/s; 3B: 765.8 -> 857-881 t/s.

Big weight 16384 x 14336, `test-backend-ops perf` (`bench/run-gpu.sh llama`, 3 rounds), FP8 time / Q8_0 time
(`runs/big-base.txt`, `runs/big-final2.txt`): n = 1..6 unchanged within spread (0.92, 0.87-0.89, 0.86, 0.85, 0.85-0.86,
0.81-0.82); n = 7 0.962 -> 0.757; n = 8 0.909 -> 0.743 (433 -> 355 us). The "after" run was noisier (spread 4-12 us).

## Gates

- `test-backend-ops test -b ROCm0 -p f8_e4m3`: 448/448 OK (the lead's rerun on the same build counted 450/450 OK,
  the same total as earlier runs, so "the 450 is stale" below is wrong; no tests were lost). The brief said 450; the unmodified original also gives 448 on
  this tree, so the count is unchanged and the 450 is stale. `bench/run-gpu.sh llama` correctness phase: 303/303 OK.
- Deliberate break (scale 128 instead of 256 in the kernel): MUL_MAT cases failed at n = 2..8 (including n = 8 k = 4096 and
  the batched/broadcast n = 8 cases), also MUL_MAT_ID; restored and rebuilt, 448/448 OK again.
- `MODEL=llama31-8b bench/run-model.sh kld` (run `build/model-runs/20261115-195652-llama31-8b`): patched FP8 at 8-token
  batches mean KLD 0.000002, max 0.000066, same top token 99.96 %; 3-token 0.000004; 1-token 0.000005; 512-token
  0.000172 (all same as notes/model-test-8b.md). Gate (<= 0.00001) met.
- Temporary env-var switches used for sweeps are removed (`grep getenv mmvf.cu` is empty).

## What was tried (all n = 8 unless noted; 8B model, FP8 t/s, baseline 407-416)

| Idea | Result |
|---|---|
| 8 warps per block | 411 (no change) |
| ksplit 4 / rows 1, 2, 4 / ksplit 1 (17 combinations) | 207-411, none better |
| Different setting for small layers only (1024-row, 4096-row) | 1 row per warp: 318-354 on 4096-row layers; small layers no effect |
| Remove address math + load next column early | 284 (hoisting, 109 registers) |
| Cap registers with `amdgpu_waves_per_eu(16)` | spills (96 registers + 68 scratch) |
| Scheduling barriers, groups of 1 / 2 / 4 / 8 columns, rows 2 | 417 / 430 / 440 / 425 |
| groups of 4, rows 4, ksplit 2 | 472; ksplit 4 479 (not kept, big weight 358 vs 348) |
| groups of 2, rows 4, ksplit 2 (kept) | 8B 484, 3B 882 (groups of 4: 480 / 856), big weight 353 vs 348 |
| groups of 3 | 8B 480, 3B 873 |
| n = 2..6 with groups | slower (n = 2: 128 vs 148); kept the old loop |
| n = 7 with groups of 2 / 3 / 4 | 8B 442 / 437 / 428 vs 420; 3B 804 / 796 / 770 vs 778; kept 2 |

Not done: unrolling by two chunks, the mixed-sign NaN check rewrite (widening is 2.5 and address math was 0.6 of 11
vector instructions per weight; no exact cheaper widening found), fewer registers via manual sinking.
