# Test plan (draft) for the FP8-in-registers GEMV and GEMM prototype

Status: drafted by a Sonnet sub-agent (read-only), checked against the code and amended by the lead (Opus) on
2026-11-15; the "Lead review" section below overrides the body where they differ. Nothing in B-D has run yet (GPU
testing paused by the owner). Labels: READ = seen in a file
(path:line), INFERRED = my reasoning, UNKNOWN = not settled. llama.cpp paths are under `build/llama.cpp` (commit 6d3568e31).

Done criterion (owner): match llama.cpp Q8_0 speed within a few percent at every batch size on the 16384 x 14336 weight,
and match llama.cpp's FP8 results to rounding error.

Terms: weight W is M rows x K columns (M=16384, K=14336 for the main shape), n is the batch (columns of activations Y).
In `test_mul_mat(type_a, type_b, m, n, k)` the weight rows are `m` and the shared length is `k` (test-backend-ops.cpp:5494).

## Lead review (overrides the body below)

- Kernels exist now: `proto/gemv/fp8_gemv.hip` (decode, variants A exact F32-activation and B FP16-activation, R rows
  per wave) and `proto/gemm/fp8_gemm.hip` (prefill WMMA). Both compare against a CPU reference and print NMSE and
  max |err| / sum|w*a|, which are gates G2 and G1. Both refuse to run unless device 0 is gfx1100 (the integrated GPU,
  gfx1036, is device 1 today).
- A1-A3 are done by `proto/widen_check.cpp`: the shift widening equals `ggml_get_type_traits(GGML_TYPE_F8_E4M3)->to_float`
  for all 254 non-NaN codes in every byte position of all 2^32 words. Mutation shown red (odd-lane mask 0x7F807F80:
  4261412864 bad bytes). The 256-code table alone did not catch that mutation; keep the word sweep.
- E1, decided by the owner: decode AND prefill, "match Q8_0 within a few percent at every batch size". The working
  number is 3 % on median time (the draft's proposal), pending the owner's confirmation.
- E2: per-tensor scale is applied in the epilogue (folded with the 2^8 from the widening). Per-row / block scales are
  out of scope for the prototype.
- E3: the prototypes reject ragged shapes (decode K % 16; prefill M % 128, K % 32; prefill N is padded). Ragged K and
  unaligned pointers (B2) apply to the real build, not the prototype.
- E5, settled from code (READ): on HIP few-row MMQ is compiled out (`mmq-few.cu:871-874`), so Q8_0 on gfx1100 runs
  MMVQ for n <= 8 (`mmvq.cuh:3`; RDNA has no type-specific branch in `ggml_cuda_should_use_mmvq`, INFERRED from reading
  `mmvq.cu:315-420`) and MMQ for n > 8 (`mmq.cu` `ggml_cuda_should_use_mmq`, RDNA3 branch: default true for Q8_0).
- B1: prototype inputs are random bytes over all non-NaN codes (full +/-448 range) and N(0,1) activations; add the
  llama.cpp-style uniform [-1, 1] data for G3 comparisons.
- Still open for the owner: E4 (F16 activations acceptable if variant B wins?) and E10 / NaN policy (0x7F/0xFF give
  NaN as llama.cpp does, at extra cost, or +/-480 as the shift gives).

## A. Correctness tests that need no GPU

A1. Exhaustive widening (256 codes), CPU only, for every code c in 0..255 compare three things:
 - model 1: the formula in `ggml-cuda/common.cuh:904-917` (READ), copied into the test;
 - model 2: `ggml_f8_e4m3_to_fp32` in `ggml/src/ggml-impl.h:576` (READ; the host version, same logic);
 - model 3: our planned register trick, written as a plain C function of the bit pattern (for example E4M3 to FP16 by shifts, with
   the 2^8 scale deferred). Compare after converting to float, bit for bit.
 Expected values worked out from the format, not from the code: normal = (1 + M/8) * 2^(E-7), subnormal (E=0) = M * 2^-9,
 sign bit flips sign, 0x7F and 0xFF are NaN (READ: `common.cuh:905-907`; ggml-impl.h:578-581 says the sign "need not be preserved").
A2. What the code must show (all READ unless noted):
 - 0x7F and 0xFF both give NaN; llama.cpp returns a positive NaN for both (`return NAN`), so test "is NaN", never a NaN bit pattern.
 - 0x00 gives +0, 0x80 gives -0 (a sign test with `signbit`, not `== 0`).
 - Subnormals 0x01..0x07 give 2^-9 .. 7*2^-9; the smallest, 2^-9, must survive in FP16 (FP16 minimum subnormal is 2^-24: INFERRED, so exact).
 - Largest finite is 0x7E = 448, and there is no infinity code in E4M3FN (INFERRED from the format; the encoder clamps to 0x7E, `common.cuh:937-938`).
 - Count: 254 finite codes + 2 NaN codes = 256, of which 2 are zeros.
A3. Exactness claim: every finite E4M3 value is exactly representable in FP16 (3 mantissa bits, exponent range inside FP16's). Test by
 round-tripping each of the 254 values through a CPU FP16 conversion (use a software conversion, not the GPU one) and requiring equality.
A4. CPU model of the widen path: after widening to FP16, check the product-by-activation and accumulation order the kernel will
 use, using float64 as truth. This produces the numbers that set the tolerances in section B (see B3). It needs the kernel's accumulation
 structure, so it can only be finished once the kernel's shape is chosen (UNKNOWN until then).
A5. The reference matmul used by every GPU test (CPU, float64 accumulation of the exact E4M3 values times F32 activations). Test it against
 a hand-computed 4 x 4 case, and against llama.cpp's CPU backend result on a small shape (the CPU F8 path is `ggml-cpu.c:302-307`, READ:
 vec_dot with F32 activations, so it is the same arithmetic as the GPU F8 path).
A6. Encoder side is out of scope for the kernels (weights are only read). If the harness quantizes F32 to E4M3 itself, test it against
 `ggml_fp32_to_f8_e4m3` (`ggml-impl.h`, saturating, round to nearest even) over all 2^16 top-halves of F32, CPU only.

## B. GPU correctness tests (to run later, under the section C commands for lock and memory cap)

### B1. Inputs
 - Random: weights uniform in [-1, 1] quantized to E4M3, activations uniform in [-1, 1]. This matches llama.cpp's own test data
   (`init_tensor_uniform` default range [-1, 1], test-backend-ops.cpp:78; F8 goes through `ggml_quantize_chunk`, :115 and :136, READ).
 - All-codes weights: each row filled by cycling all 254 finite codes with a row-dependent phase, so every code meets every K slot class.
 - Cancellation: for each row, activations chosen so the exact sum is about 0 and the terms are large (probes the WMMA mixed-sign error).
 - Wide range: activations scaled by 2^-12, 1, 2^12 (checks F16 overflow and underflow if the kernel narrows activations to F16).
 - NaN code: one NaN code (0x7F, later 0xFF) in one row; every output of that row must be NaN, all other rows unchanged. A zero activation
   still gives NaN, because NaN * 0 = NaN (INFERRED; matches `ggml_cuda_mad`, `common.cuh:773-780`).
 - Fixed seeds, printed. Three repeated runs of the same input must be bit-identical (the WMMA probe found it deterministic:
   `matrix-arithmetic.md` section 3.2, READ).

### B2. Shapes
Decode GEMV (batch n = 1, 2, 3, 4, 5, 6, 7, 8, each separately; llama.cpp's F8 decode is used for n <= 8: `mmvf.cuh:3`, `mmvf.cu:869-905`, READ):
 - M in {16384, 4096, 152064 (LM head size from survey section 8)} plus ragged: 1, 2, 31, 33, and r*RPB-1, r*RPB, r*RPB+1 where RPB is the
   rows-per-block the kernel picks (UNKNOWN until chosen), plus one prime near 4096 (4093).
 - K in {16, 32, 4096, 14336} plus ragged: values that are not a multiple of the kernel's K tile and not of 16: 4100, 14338, 100, 2, and
   one just above a tile edge. llama.cpp's F8 path needs K even and 2-byte alignment (`mmvf.cu:875, 883`, READ); our 16-byte loads need
   more, so decide whether the kernel handles ragged K or refuses it (open question E3). If it refuses, test that it refuses cleanly.
 - Weight pointer alignment: base at 16-byte alignment, and (if supported) offset by 1, 2, 4 bytes as llama.cpp does (test-backend-ops.cpp:11934-11940, READ).
Prefill GEMM (WMMA, n = 64..4096):
 - n in {64, 65, 127, 128, 129, 255, 256, 511, 512, 513, 1000, 1024, 2048, 4095, 4096}: below, on, and above the tile size; not multiples of 16.
 - M in {16384, 4096, 4097 (if supported)}, K in {14336, 4096, 4112}. K must be a multiple of 16 for the 16x16x16 instruction (INFERRED);
   ragged K needs padding or a tail path (open question E3).
 - Big shapes are checked on a subset: all n columns for 256 sampled rows (including the first and last 16 rows and rows around tile edges), because a
   full float64 reference at n=4096 is about 2 TFLOP on the CPU (INFERRED arithmetic: 2*16384*14336*4096).
 - Every shape that appears in section C must appear here first.

### B3. Reference and tolerance (derived, not guessed)
Reference R: float64 sum of (exact E4M3 value) * (F32 activation). Let S = sum_k |w_k * y_k| for the same output (float64).
Per-element gate G1 (catches a wrong row or column, which an NMSE gate hides):
  |d - R| <= eps * S, with eps = eps_act + eps_acc + eps_wmma.
 - eps_act: 0 if activations stay F32. If the kernel rounds activations to F16 (round to nearest) the relative error per element is at most 2^-11,
   so eps_act = 2^-11 = 4.9e-4 as a worst-case bound (INFERRED from the 10-bit stored mantissa).
 - eps_acc: F32 accumulation. Worst case is (L + log2 P) * 2^-24, where L is the longest serial chain of additions in the kernel and P the
   number of parallel partial sums combined in the reduction (INFERRED, standard bound). Fill in L and P from the kernel (UNKNOWN until written).
 - eps_wmma (prefill only): the F16 WMMA error with mixed signs is about u * 2^(E-24), u about 1 to 1.4 on average, over 99 % within 4 units, where E is the
   exponent of the largest term (`errors-after-quantizing.md:56`, `matrix-arithmetic.md` headline, READ). Use 4 * 2^-24 per 16-term group as the
   bound, applied to that group's own sum of |terms|. The true maximum is not in what I read (UNKNOWN): measure it on cancellation inputs and raise the bound if needed.
 Because a worst-case bound is loose by about sqrt(K) for random data, G1 is a safety gate, not a precision claim.
Statistical gate G2 (the real precision claim): NMSE of output against R, using the same formula as llama.cpp, `nmse()` at test-backend-ops.cpp:326 (READ).
 Threshold = 4 x the NMSE that the CPU model of our own arithmetic gives on the same inputs (A4). Reason: NMSE depends on the data, so the number
 comes from simulating the intended arithmetic. Order of magnitude to check against (INFERRED): activation rounding to F16 gives NMSE of about
 (2^-11)^2 / 3 = 8e-8 at most; F32 activations with F32 accumulation give around K * 2^-48, below 1e-10.
Compatibility gate G0: NMSE <= 5e-4 against llama.cpp's CPU result, the threshold llama.cpp uses for `MUL_MAT` (`test-backend-ops.cpp:5519-5522`, READ; the
 default elsewhere is 1e-7, :1266). Note 5e-4 is far looser than G2, and one completely wrong row in 16384 gives NMSE of about 1/16384 = 6e-5, which passes G0.
 That is why G1 and G2 are also required.
Gate G3, against llama.cpp's own F8 output on identical inputs (same weights bytes, same F32 activations, shapes where llama.cpp supports them: K even, n <= 8
 on its GEMV, and its dequantize + hipBLAS path for n > 8): compute nmse(ours, llama) and require <= (sqrt(T_ours) + sqrt(T_llama))^2, where T_llama is llama.cpp's
 own NMSE against R on those inputs. (From the triangle inequality for the 2-norm: INFERRED.) Method: a standalone program loads the llama.cpp
 library, or use `test-backend-ops` output dumping if available (UNKNOWN whether it has one); do not change llama.cpp for this.
Per-scale: llama.cpp applies a per-tensor scale outside the weight tensor (`mmvf.cu:74, 101, 397`, READ: fusion `x_scale`); the test weights here have none.
 Whether the prototype must apply a scale in the epilogue is open question E2.

## C. Performance tests (to run later)

C1. Shapes and batch sizes. Main shape: weight 16384 x 14336, F32 activations, F32 output.
 - Decode: n = 1, 2, 3, 4, 5, 6, 7, 8. Prefill: n = 64, 128, 256, 512, 1024, 2048, 4096.
 - Secondary (from survey section 8): 4096x4096, 4096x14336, 14336x4096, 152064x4096 at n = 1, 4, 8, 512, for the same weights; these are small enough to sit in cache, so see C3.
C2. Baselines (all on the same card, same session, same lock):
 - llama.cpp Q8_0, F16, F8_E4M3 via `test-backend-ops perf -b ROCm0 -o MUL_MAT -p type_a=...` with the out-of-cache patch
   `bench/llama-cpp-out-of-cache-shape.patch`. Always pass `-b ROCm0`: the log shows a second device (the CPU's integrated GPU, `build/perf-mulmat-big.log:3`).
 - The patch only has n = 1, 4, 8, 512 (READ, patch lines 13-18). Q8_0 at n = 2, 3, 5, 6, 7 and n = 64..4096 needs the patch extended (a local edit to the patch file, not to
   the repo's source). Note the dispatcher may route some n to the few-row MMQ kernel instead of MMVQ (`ggml-cuda.cu:2490-2497`, READ), so the Q8_0 baseline at each n is
   whatever llama.cpp picks; record which kernel it was (UNKNOWN for gfx1100).
 - Existing numbers (READ, `notes/go-no-go-1.md`, `build/perf-mulmat-big.log:10-13`): Q8_0 346 us at n=1, 386 at n=4, 483 at n=8, 4202 at n=512; F8 1099, 1210, 1669, 5147; F16 636, 757, 770, 4245. One run each, no spread yet.
 - Own roofline: `bench/bandwidth.hip` read ceiling 754 GB/s (READ). For our kernel, ideal time for 234,881,024 FP8 bytes is 311 us at that ceiling (INFERRED arithmetic: 16384*14336 = 234.9 MB).
   Q8_0 has 1.0625 bytes per weight (249.6 MB), so a Q8_0 at 96 % of the ceiling is 346 us; an FP8 kernel at the same 96 % would be about 324 us, about 6 % faster.
C3. Infinity Cache trap (80 MB). A weight smaller than the cache, or re-read right after itself, shows impossible speeds (READ: Q8_0 "reading" 1737 GB/s on 4096x14336, go-no-go-1.md).
 - Main shape at 235 MB is above the cache; llama.cpp Q8_0 measured 721 GB/s there, below the ceiling, so it is clean (READ).
 - For our own harness: keep a pool of weight copies whose total size is at least 4 x 80 MB = 320 MB and rotate through them each launch (so a weight's reuse distance exceeds the cache).
   For the main shape that means 2 copies (470 MB). For 4096x4096 (16 MB) it means 20 copies. Memory is fine on a 20 GB card.
 - Reject any result above 1.03 x 754 GB/s of weight bytes per second at decode: that is a cache hit, not a measurement. Print GB/s next to every time.
 - Prefill is compute-bound (F16 at n=512 is about 57 TFLOPS from the log numbers: 469.76 MFLOP*512/4245 us, INFERRED), so cache does not apply, but still use the pool.
C4. Timing method. Warm up for at least 1 s of GPU work (clock ramp) and 20 launches. Then 15 timed sets of 200 back-to-back launches each (hipEvent around each set), report per-launch
 median, min and max, and (max-min)/median as the spread. Before and after, record the GPU clock and memory clock from sysfs (read only). One process, one kernel at a time, no overlap with the baseline runs.
 Cross-check the harness: run a plain read-only kernel in it; it must reproduce about 754 GB/s, and a F16-weight GEMV in our harness must land near llama.cpp's F16 636 us. If not, the harness is wrong, not the kernel.
 Take baselines in the same session, interleaved (our kernel, Q8_0, our kernel, Q8_0), to cancel thermal and clock drift.
C5. What counts as "within a few percent" (needs the owner's number, E1). Proposal: our median time <= Q8_0 median time * 1.03 at every n, AND the gap exceeds the measured spread to be called a
 loss or win. Before judging anything, take 10 repeats of the Q8_0 baseline at each n to learn the spread (go-no-go-1.md "Not measured"). If spread is above 3 %, the 3 % gate is meaningless and must be widened or the method fixed.
 A kernel faster than Q8_0 passes. Report the ratio ours/Q8_0 for each n in one table.
C6. Commands (from HANDOFF.md rules, READ; combine them as follows, one job at a time, after `free -g` shows at least 8 GB available):
 `flock /home/jonathans/Projects/wronggpu/build/rdna3-analysis/gpu-bench.lock systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- nice <command>`
 Headless, no window. Scratch and logs under the project folder, never `/tmp`. Run no more than two heavy jobs at once (machine rules); prefer one for timing.

## D. Tests that must be able to fail (break on purpose, confirm red, restore, confirm green)

For each, the mutation goes in a scratch copy, the named test must go red, then the restore must go green. Record both runs in the change summary.
 D1. A1/A2: flip the NaN rule so 0x7F maps to 448; flip subnormal scale 2^-9 to 2^-8; drop the sign bit for 0x80. Expect A1 red in all three.
 D2. A3: truncate instead of round when going to FP16 in the CPU test (or use bf16). It must not matter for E4M3 (exact), so this test checks that A3 stays green; a mutation that must stay green proves the test is looking at the right thing.
 D3. B3 G1: make the kernel skip the last row of each tile (M ragged), or skip the K tail. G0 (5e-4) is expected to stay GREEN for one wrong row out of 16384 (about 6e-5); G1 and G2 must go red. This proves why G1 exists.
 D4. B2 ragged K: read K rounded down to a multiple of the tile. G1 red on ragged K shapes, green on tile-multiple shapes.
 D5. B1 NaN: replace NaN handling with a clamp to 0. The NaN-row test must go red.
 D6. B3 G2: round activations to bf16 instead of F16 (relative error 2^-8, 8x larger). NMSE should rise by about 64x and trip G2 while staying below G0.
 D7. Prefill tile edge: shift a WMMA B-fragment index by one K slot. G1 red on all prefill shapes.
 D8. n loop: in the decode kernel, hardcode n=1 columns. Test n=2..8 red; n=1 green.
 D9. Determinism: add a data race (remove one barrier or reduction sync) in a scratch copy; the three-run bit-identical check should go red at least on one large shape (a race may be intermittent: run it 20 times and say so).
 D10. Perf harness checks: (a) run on a weight small enough for the cache with the pool disabled; the "above 1.03 x 754 GB/s" rejection must trip. (b) Insert a 5 % busy-wait into our kernel; the ratio table must show it and C5 must go red. (c) Time an empty kernel and confirm the harness reports launch-overhead scale, not hundreds of microseconds.
 D11. Compare-to-llama (G3): feed our kernel a different weight byte than llama.cpp gets (flip one byte); G3 must go red.

## E. Open questions the plan cannot settle from the code

 E1. What number is "within a few percent" (1 %, 3 %, 5 %), and is it on median time? And is "every batch size" decode only (n = 1..8) or does it include prefill n = 64..4096? Matching Q8_0 prefill (about 57 TFLOPS at n=512, INFERRED) with a WMMA kernel written from scratch is a much larger job than decode.
 E2. Does the prototype apply a per-tensor scale (llama.cpp's fused `x_scale`, `mmvf.cu:101, 397`)? Per-row or 128-block scales? The task text says none.
 E3. Ragged K and unaligned weight pointers: handled, or rejected with an error? llama.cpp's F8 decode accepts any even K with 2-byte alignment (`mmvf.cu:875, 883`). Rows with a stride other than K (views) are also unspecified.
 E4. Activation precision: does the decode kernel keep F32 activations (llama.cpp semantics, tolerance near 1e-10) or narrow to F16 for `v_dot2_f32_f16` (survey section 8 plan, NMSE about 8e-8)? The tolerance changes by about three orders; this must be fixed before B3 numbers are final.
 E5. Which kernel does llama.cpp route Q8_0 to at n = 2..7 and 64..4096 on gfx1100 (MMVQ, few-row MMQ, MMQ, or hipBLAS)? This sets the prefill baseline and is not in the logs I read.
 E6. Spread of the baseline. Not measured yet, so the pass margin in C5 is provisional.
 E7. Does `test-backend-ops` have a way to dump outputs for G3, or must a small standalone program use the llama.cpp libraries? (UNKNOWN; I did not search it.)
 E8. Maximum (not typical) WMMA error with mixed signs on RDNA 3: the carry-over files give means and a 99 % figure only (B3 eps_wmma).
 E9. Which output type the prototype's GEMM produces (F32 as in llama.cpp, or F16), and whether n > 8 batches share one weight read (they must, to compare fairly with the dequantize-then-hipBLAS path).
 E10. NaN sign: llama.cpp returns a positive NaN for both NaN codes; any NaN is accepted by the plan's tests. Confirm that is the intended match.
