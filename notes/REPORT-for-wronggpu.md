# rdna3-fp8: report for the WrongGPU agent

Date: 2026-11-15. Written by the Claude session that ran the rdna3-fp8 project (repo `/home/jonathans/Projects/rdna3-fp8`,
local only, branch `feat/go-no-go-measurement`, nothing pushed). This project started from material WrongGPU handed over
(`carry-over/`: the exact FP32 -> E4M3 rounding and the RX 7900 XT instruction measurements from the WrongGPU RDNA 3
analysis, issue #28). This report returns what was learned. Every number below was measured on this machine's
RX 7900 XT (gfx1100, 20 GB) unless marked INFERRED; the detailed notes it summarises are listed at the end.

## 1. Summary

- Goal: make FP8 (E4M3) models cheap on RDNA 3, which has no FP8 instructions, by keeping FP8 bytes in VRAM and widening
  them in registers, inside a real runtime (llama.cpp) and PyTorch.
- Result in llama.cpp (Llama 3.2 3B Instruct FP8, patched llama.cpp vs Q8_0 of the same weights):

| What | FP8 vs Q8_0 | Before this work |
|---|---|---|
| Generation, one stream | about +10 % (157.8 vs 145.0 t/s) | FP8 2.2x slower than Q8_0 (66 t/s) |
| 2-7 parallel streams | +1 % to +7 % | FP8 2.2-2.8x slower |
| 8 parallel streams | -10 % (open) | |
| Prompt processing | +12 % to +20 % (pp512 8545-8962 vs 7660-7678 t/s) | NaN output (bug, section 4) |
| GPU memory | -5 % (4.18 vs 4.37 GB peak) | |
| Answer quality | same as Q8_0; closer to exact than Q8_0 | garbage ("GGGG...") on any prompt > 8 tokens |

- Unmodified llama.cpp produces NaN for every FP8 model on RDNA 3 as soon as more than 8 tokens are processed at once.
  Root cause found and fixed locally (section 4).
- In PyTorch 2.13 (ROCm 7.1) every native FP8 matmul raises on gfx1100; a small extension running our decode kernel is
  2.3-2.6x faster than PyTorch's own FP16 `F.linear` at half the weight memory.
- Everything lives as a patch file (`bench/llama-cpp-fp8-mmvf-hip.patch`, 470 lines, 4 files) applied to a local
  llama.cpp clone; nothing was committed inside llama.cpp (its AGENTS.md forbids agent commits) and nothing was sent
  upstream. Whether to offer it upstream is the owner's decision.

## 2. Setup and rules that applied

- llama.cpp `6d3568e31` (tag b12408) cloned to `build/llama.cpp` (patched) and a worktree `build/llama.cpp-base`
  (unpatched); HIP build `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`. Needed `hipblas-devel` and `rocblas-devel` (installed
  with the owner's sudo password, 2026-11-15). ROCm/HIP 7.1.1 from the distribution.
- Every GPU-timed run took the shared lock `/home/jonathans/Projects/wronggpu/build/rdna3-analysis/gpu-bench.lock`
  (one command per acquisition, so WrongGPU's runs got turns), ran under `systemd-run --user --scope -p MemoryMax=...
  -p MemorySwapMax=0 nice`, and checked available memory first (8 GB floor, stop below 3 GB).
- The machine has a second ROCm device: device 0 = RX 7900 XT (gfx1100), device 1 = the Ryzen iGPU (gfx1036).
  llama.cpp and PyTorch both see it. All runs used `HIP_VISIBLE_DEVICES=0`; the standalone programs refuse to run unless
  device 0 is gfx1100. Without this, llama.cpp may split layers onto the iGPU.
- One worker (Sonnet) was killed by a system-wide out-of-memory event caused by a browser tab using 11.2 GB, not by
  this project. Its partial work was recovered and re-measured by the lead.

## 3. The core technique: E4M3 -> FP16 by bit shifts

One 32-bit word of four E4M3 bytes b0..b3 becomes two FP16 pairs with 3 ALU ops per pair:

```
even = ((x << 8) & 0x80008000) | ((x << 7) & 0x3F803F80)   // halves (b0, b2)
odd  = ( x       & 0x80008000) | ((x >> 1) & 0x3F803F80)   // halves (b1, b3)
```

Each half equals the E4M3 value times 2^-8 exactly, subnormals included (E4M3 subnormal m * 2^-9 lands on FP16
subnormal m * 2^-17). The 2^8 goes into the output scale. The pair order is k0, k2, k1, k3, so the activations must be
packed in the same order; a dot product does not care.

- Checked against llama.cpp's own CPU conversion (`ggml_get_type_traits(GGML_TYPE_F8_E4M3)->to_float`) for all 2^32
  words, every byte position: exact for all 254 non-NaN codes, no leak between bytes (`proto/widen_check.cpp`). A
  mutant with the odd mask widened to 0x7F807F80 fails on 4,261,412,864 bytes; the 256-code table alone did NOT catch
  it, so test whole words.
- NaN codes (0x7F, 0xFF) widen to +-480, not NaN. Per-word NaN test, exact over all 2^32 words:
  `((x & 0x7F7F7F7F) + 0x01010101) & 0x80808080` is non-zero iff some byte is 0x7F/0xFF. OR it into a per-row flag and
  write NaN rows at the end; that equals llama.cpp's semantics because NaN times anything is NaN. Cost about 0.75 ALU
  ops per weight, no branch.
- For FP32 activations the widened half feeds `fmaf(float(half), a, acc)`, which the compiler turns into
  `v_fma_mix_f32` (FP16 x FP32 into FP32, one per weight per column, not dual-issuable). For FP16 activations,
  `v_dot2_f32_f16` does two weights per instruction.

## 4. Bug found in unmodified llama.cpp: FP8 models give NaN on RDNA 3

- Symptom: perplexity NaN for every chunk; generation prints only "GGGG..." for any prompt longer than 8 tokens.
- Cause: for more than 8 columns, `ggml_cuda_mul_mat_cublas` converts F8 weights to F16 and, on GPUs with fast F16,
  calls hipBLAS with `CUBLAS_COMPUTE_16F` and F16 output. Only Volta, RDNA 4 and CDNA get F32 output
  (`prefer_f32_output` in `ggml_cuda_mul_mat_cublas_impl`). FP8 checkpoints store unscaled weights (up to 448) and apply
  the per-tensor scale (about 1e-3 here) in a later MUL, so the F16 output overflows 65504 -> inf -> NaN.
- Evidence (unpatched, same file, 4 chunks): CPU 9.7036; GPU with 8-token batches (mmvf, F32 accumulation) 9.7038;
  GPU 512-token batches NaN; GPU with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` 9.7032. Weights: no NaN codes, scales finite.
- Why llama.cpp's tests miss it: their F8 test weights are uniform in [-1, 1]. Also their MUL_MAT tolerance
  (NMSE 5e-4) passes a 1/256 scale error; a 2x error is needed to fail.
- Fix 1 (fallback, one line): F8 weights get F32 output in that function. Correct, but prompts drop to 1850 t/s on
  RDNA 3 (hipBLAS F32-output routines are slow here: 2306 us vs 509 us at 8192x3072x512).
- Fix 2 (scale folding): convert F8 -> F16 with the per-tensor scale applied at llama.cpp's existing MUL_MAT + scale
  fusion hook, then the normal fast F16 path. 6635 t/s, correct, but F16-accumulation precision (KLD 0.0011).
- Fix 3 (final): a dedicated FP8 WMMA kernel at the same hook (section 6). Fixes 1 and 2 remain as fallbacks for shapes
  it does not take.

## 5. Decode (1-8 tokens at once)

- Baseline cause: llama.cpp's FP8 decode loads 2 bytes per lane and converts each byte with `ldexpf` in software; it
  reached 214 GB/s (28 % of the card's 754 GB/s read ceiling) at batch 1, vs Q8_0 721 GB/s and F16 739 GB/s.
- Batch 1 path (inside llama.cpp's `mul_mat_vec_f`): 16-byte loads per lane, shift widening, `fmaf` into FP32,
  NaN through the block sum. 312-322 us vs Q8_0 338-366 us on a 16384 x 14336 weight.
- Batch 2-8: a separate kernel `mul_mat_vec_fp8_rows<ncols, rows_per_warp, ksplit>`:
  - several rows per warp so one activation load serves several rows (the original kernel reloads activations per row,
    which made n >= 4 slow);
  - `__builtin_nontemporal_load` on the weights (`global_load_b128 ... slc dlc`): weights are read once, activations
    are reused; without the hint the streaming weights evict the activations. This alone took n = 7, 8 from 1.04-1.10x
    to 0.93x Q8_0's time on the big weight;
  - `ksplit` = 2 warps per row group splitting the row length (partial sums through LDS): needed for small model
    layers (1024 x 3072), where 4 rows per warp left only 64 blocks for 84 CUs;
  - final setting from sweeps in the model and on the big weight: 4 rows per warp for 2-7 columns, 2 for 8, ksplit 2.
- Things that did NOT help: `__launch_bounds__(128)` (no effect); converting each widened weight to FP32 so the products
  could dual-issue as `v_dual_fmac_f32` (the compiler pairs few; 2-3 %, within noise; 1.6-1.9x slower at 1 row/warp).
- Correctness: KL divergence vs unmodified llama.cpp's exact F32 decode path (8-token batches as reference):
  0.000002-0.000004, same top token 99.9 %, at 1-, 3- and 8-token batches; Q8_0 is 0.000605 on the same reference.

## 6. Prompt processing (more than 8 tokens): FP8 WMMA GEMM

- Kernel `mul_mat_fp8_wmma` (patch, `fp8.cu`, RDNA 3 only via `#if defined(RDNA3)`): tile 128 rows x 128 columns x 64 K,
  8 waves (4 along rows, 2 along columns), one LDS buffer; E4M3 widened to F16 while staging; F32 activations rounded to
  F16 while staging (no separate pass); `v_wmma_f32_16x16x16_f16_w32` with F32 accumulation; activations as the A
  operand and weights as B, so each accumulator lane runs along the weight rows and stores are contiguous; columns past
  N clamped on load; NaN rows; scale read from the scale tensor in the epilogue. Used for more than 8 columns when
  M % 128 == 0, K % 64 == 0, tensors contiguous and 16-byte aligned.
- How it got there: a first prototype (`proto/gemm`, v1) reached 37-41 TFLOPS on 16384 x 14336 and only 14-38 TFLOPS on
  the 3B shapes; register prefetch + double buffering (v2) gave 7-13 %. A Sonnet worker then built a configurable
  version (tile sizes, waves, split-K, fused activation conversion, rasterization, about 50 variants) with a hipBLAS
  baseline in the same harness. Its best general configuration, re-timed by the lead at N = 512 (us, hipBLAS F16 vs
  ours without activation conversion): 8192x3072 512-534 vs 355-357; 3072x3072 156-157 vs 152-154; 1024x3072 60-62
  vs 66-68; 3072x8192 393-395 vs 388-391. hipBLAS here also excludes llama.cpp's per-call weight conversion.
- In the model: pp512 8545-8962 t/s vs Q8_0 7660-7678 (+12-17 %); pp2048 with `-ub 2048` 9213-9225 vs 7563-7657
  (+20 %). KL divergence of prompt processing vs the exact reference 0.000153 (scale folding was 0.001076, the F16
  model on llama.cpp's own F16 path 0.002061, Q8_0 0.000605). Perplexity 9.4626 (Q8_0 9.4743, F16 9.4942).
- Open: one tile configuration for all shapes; the small K/V layer is about 10 % behind hipBLAS in isolation.

## 7. End-to-end model test

- Model: RedHatAI/Llama-3.2-3B-Instruct-FP8 (compressed-tensors `float-quantized`, one scale per tensor, Llama 3.2
  Community License), 4.4 GB, in `models/` (gitignored). llama.cpp's converter keeps FP8 only for one-element scales;
  per-channel ("FP8-dynamic") and 128-block checkpoints are dequantized at conversion. Its `naive-quantized` path
  (e.g. RedHatAI Llama-3.1-8B-Instruct-FP8) would hit an assert on reading (INFERRED from the code, not run).
- The checkpoint ships `lm_head` in BF16 (788 MB read per token). `proto/gguf_output_q8.py` makes a variant with a Q8_0
  output layer (all other tensors byte-identical) so the comparison measures the FP8 weights. Note: `GGUFReader`
  exposes BF16 data as raw uint8, two per value.
- Real prompts (greedy): "how does a refrigerator keep food cold" and "write a Python prime check": patched FP8 and Q8_0
  replies are byte-identical and correct; unmodified llama.cpp on the FP8 file prints only "GGGG..."
  (`notes/prompt-test-1.md`).
- GPU memory (peak during a 4.5 KB prompt, from the driver): FP8 with Q8_0 output 4175 MiB, Q8_0 4369, FP8 as shipped
  4533, F16 7280. Model buffers 3088 / 3256 / 3440 / 6128 MiB. FP8 saves 1/17 of the quantized bytes vs Q8_0; the new
  prompt kernel needs no temporary buffers.

## 8. PyTorch on gfx1100

- PyTorch 2.13.0+rocm7.1 (`build/torch-venv`): `torch._scaled_mm` raises ("only supported on CUDA devices with compute
  capability >= 9.0 or 8.9, or ROCm MI300+"); matmul/F.linear with an FP8 operand are not implemented; `.to()`
  conversions work. Workarounds at 16384 x 14336: FP16 weights resident 826-945 us (569 GB/s), FP8 converted per call
  4057-4519 us.
- `proto/torch/fp8_linear.py`: `FP8Linear` keeps float8_e4m3fn weights in VRAM and calls the decode prototype through
  ctypes with PyTorch's stream. PyTorch ships its own `libamdhip64.so` with the same soname as the system one; import
  torch first and both share PyTorch's copy (the script checks /proc/self/maps). 322-508 us (exact) / 330-403 us (FP16
  activations) at n = 1..8, 2.3-2.6x faster than F16 `F.linear`.

## 9. Lessons that transfer to other RDNA 3 kernel work (WrongGPU included)

- The 80 MB Infinity Cache fakes bandwidth: llama.cpp's own perf shape (4096 x 14336, about 60 MB as 8-bit) showed Q8_0
  "reading" 1737 GB/s. Use weights larger than about 4 x 80 MB, or rotate copies, for any memory-bound timing.
- Measured ceilings on this card: read 754 GB/s (median, 1 GiB, 16-byte loads), copy 603 GB/s.
- LLVM splits 32-bit bit tricks into 16-bit ops when the result feeds an FP16 vector: an empty `asm("" : "+v"(x))`
  after the widening keeps the word whole (variant B went from 8.4 to 2.6 ALU ops per weight).
- `memcpy` into a `_Float16 __attribute__((ext_vector_type(16)))` from LDS words made LLVM move each half separately
  (784 VALU per 16 WMMA); `__builtin_bit_cast` of a `uint32 ext_vector_type(8), aligned(16)` load fixes it (52).
- `__builtin_nontemporal_load` needs a native vector type (`ext_vector_type`), not HIP's `uint4`.
- WMMA f32_16x16x16_f16_w32 layout confirmed on the card: A lane%16 = row, B lane%16 = column, 16 halves of K per lane;
  D element e of lane t is row 2e + t/16, column t%16. Swapping operands (activations as A) makes D lanes run along the
  weight rows for contiguous stores.
- `v_fma_mix_f32` is what `fmaf((float) half, f32, f32)` compiles to; one per issue slot. Forcing FP32 conversion to
  get `v_dual_fmac_f32` did not pay.
- Small matrices leave the card idle: count blocks against 84 CUs (42 WGPs). Splitting K across warps (decode) or
  per-shape tiles (GEMM) is where the gains were.
- A loop-instruction counter on the `-save-temps` assembly (`proto/gemv/loop_isa.py`) caught every codegen problem above
  before any GPU time was spent; pick loops by their math instruction, since a kernel can hold several loops.
- Tests must be able to fail: llama.cpp's NMSE 5e-4 gate passed a 1/256 scale error; break the code on purpose and
  confirm red before trusting green. A per-output check (|error| / sum |w*a|) catches a single wrong row that NMSE hides.
- A kernel that looks instruction-bound may be memory-bound: at 11 ALU ops per weight the n = 8 decode kernel used about
  40 % of the instruction rate (INFERRED from the instruction count, the measured time and the carry-over issue rate);
  the fix was a cache hint, not fewer instructions.

## 10. Files

- Patch: `bench/llama-cpp-fp8-mmvf-hip.patch` (apply to llama.cpp 6d3568e31: `git -C <llama.cpp> apply <patch>`);
  benchmark-only patch for out-of-cache shapes: `bench/llama-cpp-out-of-cache-shape.patch`.
- Scripts: `bench/run-gpu.sh` (prototype correctness and timing, llama.cpp op timing), `bench/run-model.sh` (model
  perplexity, KL divergence, llama-bench, llama-batched-bench), `bench/bandwidth.hip`.
- Prototypes: `proto/gemv/fp8_gemv.hip` (decode variants A/B/C), `proto/gemm/fp8_gemm.hip` (prompt GEMM variants,
  hipBLAS baseline; link `-lhipblas`), `proto/widen_check.cpp`, `proto/torch/`.
- Notes: `notes/go-no-go-1.md`, `notes/prototype-results-1.md`, `notes/llama-cpp-integration-1.md`,
  `notes/model-test-1.md`, `notes/prompt-test-1.md`, `notes/pytorch-baseline-1.md`, `notes/pytorch-ext-1.md`,
  `notes/test-plan.md`, `notes/survey.md`.

## 11. Open items

- 8 parallel streams: FP8 about 10 % behind Q8_0 in the 3B model (ahead on the 16384 x 14336 weight).
- Prompt kernel: per-shape tile choice or split-K for the small K/V layers.
- No independent review of the llama.cpp patch yet (offered to the owner).
- Untested: other models and sizes, RDNA 4 (the WMMA kernel is RDNA 3 only by design; RDNA 4 has native FP8 and a
  different WMMA layout), MUL_MAT_ID (MoE) shapes, Vulkan.
