# PyTorch ROCm baseline 1: FP8 weights on the RX 7900 XT (gfx1100)

Date: 2026-11-15. PyTorch 2.13.0+rocm7.1, `HIP_VISIBLE_DEVICES=0`, arch asserted `gfx1100`.
Scripts: `proto/torch/probe.py` (Part 1), `proto/torch/bench.py` (Part 2). Logs: `build/torch-runs/probe.log`,
`build/torch-runs/bench.jsonl` (gitignored). All numbers MEASURED unless labelled. Bench run held the shared GPU lock,
8 GB memory cap, `nice`; one run, 19 s. Single run only, so no run-to-run spread is known (card shared with WrongGPU).

## Part 1: behaviour, weight (16384, 14336) fp8, device 0 (MEASURED)

Same results for `float8_e4m3fn` and `float8_e4m3fnuz`, n = 1 and 8:

| Call | Outcome |
|---|---|
| `.to(float8_*)` from fp16 | works |
| `w.to(float16)`, `w.to(bfloat16)`, `* scale` | work |
| `torch._scaled_mm` (fp8 x fp8, fp16 act, bf16 act) | RuntimeError: `torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+` |
| `torch.matmul` / `F.linear`, fp16 act x fp8 weight | RuntimeError: `expected mat1 and mat2 to have the same dtype, but got: c10::Half != c10::Float8_e4m3fn` (fnuz: `...Float8_e4m3fnuz`) |
| `torch.matmul` / `F.linear`, fp8 x fp8 | NotImplementedError: `"addmm_cuda" not implemented for 'Float8_e4m3fn'` (fnuz alike) |

So no native fp8 path exists on this card: case (d) is not available. Only convert-then-FP16 works.

## Part 2: decode timing, weight 16384 x 14336, x (n, 14336), out (n, 16384) (MEASURED)

`torch.cuda.Event` per call, 20 warmup, 200 timed (20 timed for n = 512). Median us, (min-max) in brackets.
Weights 235 MB (FP8) and 470 MB (FP16/BF16) exceed the 80 MB Infinity Cache, so no rotation of weight copies
was needed (INFERRED from sizes; the repo's cache rotation patch is for smaller shapes).

| n | (a) F16 resident | (b) BF16 resident | (c) FP8 convert per call | llama.cpp Q8_0 (READ) | ours A (READ) |
|---|---|---|---|---|---|
| 1 | 826 (782-982) | 816 (778-1006) | 4457 (3944-5322) | 353 | 317 |
| 2 | 945 (802-1293) | 937 (788-1330) | 4519 (4171-5299) | 372 | 321 |
| 3 | 926 (804-1420) | 930 (801-1493) | 4148 (3927-5203) | 380 | 339 |
| 4 | 826 (784-1088) | 825 (778-1065) | 4057 (3964-5052) | 385 | 349 |
| 5 | 841 (775-1124) | 840 (783-1219) | 4084 (3942-5353) | 389 | 364 |
| 6 | 848 (802-1465) | 833 (784-1147) | 4096 (3940-4598) | 414 | 378 |
| 7 | 839 (778-1118) | 834 (789-1201) | 4097 (3945-4570) | 447 | 450 |
| 8 | 840 (789-1315) | 845 (782-1299) | 4252 (3939-4873) | 484 | 480 |
| 512 | 3980 (3685-4292) | 4334 (4109-4437) | 7205 (6718-7929) | n/a | n/a |

Quoted comparison numbers: READ notes/prototype-results-1.md (decode table, Q8_0 and variant A columns).

Effective weight read bandwidth = stored weight bytes / median time (MEASURED, n = 1 / n = 8):
- (a) F16: 569 / 559 GB/s; (b) BF16: 576 / 556 GB/s (n = 512: 118 / 108 GB/s, compute-bound regime).
- (c) FP8 convert: 53 / 55 GB/s on FP8 bytes (the call moves about 2.1 GB, INFERRED: convert reads 235 MB and
  writes 470 MB, multiply reads and writes 940 MB, GEMV reads 470 MB; at 4.1 ms that is about 510 GB/s of real traffic).
- For reference (READ prototype-results-1.md): ours A at n = 1 reads 741 GB/s of FP8 bytes; read ceiling 754 GB/s.

Ratios at n = 1 (MEASURED / READ): F16 resident 2.34x Q8_0 and 2.61x ours A; per-call convert 12.6x Q8_0 and
14.1x ours A. At n = 8: F16 1.74x Q8_0, 1.75x ours A; convert 8.8x Q8_0, 8.9x ours A.
Resident F16 needs twice the VRAM of FP8 (470 vs 235 MB per this weight), which is the reason FP8 is wanted.

## Correctness (MEASURED)

Max relative error = max|out - ref| / max|ref| over the first 64 output rows, ref in float64 on CPU from the exact
fp8 value x 0.5 (a power of two, so the scale is exact) and the activation actually passed (fp16, or bf16-rounded).
- (a) and (c): 2.2e-4 to 3.2e-4 for all n (fp16 accumulate/output rounding); identical values for (a) and (c),
  as both use the same fp16 weights and activations.
- (b): 2.2e-3 to 2.8e-3 for all n (bf16 output rounding, about 8x worse than fp16, as expected from 8 vs 11 bits).
No path is broken, so all timings above are for working results. (d) has nothing to check.

## Caveats

- Single timed run, one process; per-call min-max is wide for (a) and (b) (up to 1.8x), probably the shared card
  (INFERRED, clocks not sampled). Medians for n = 2 and 3 are noisier than n = 4..8 (945/926 vs 826-841).
- (c) as written uses two temporaries (convert, then multiply); a fused convert+scale in PyTorch eager does not
  exist, so this is the usual eager workaround and an upper bound on its cost. Not tried: `torch.compile`
  fusing the convert into the GEMM prologue (not requested).
- The harnesses differ from llama.cpp's `test-backend-ops` (hipEvent in ours, its own timing there); PyTorch
  numbers include launch overhead of F.linear (GEMM dispatch via hipBLASLt/rocBLAS, library not checked).
