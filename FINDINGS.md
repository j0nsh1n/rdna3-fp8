# Findings: FP8 (E4M3) on RDNA 3

Technical summary of the rdna3-fp8 work. All numbers were measured on a Radeon RX 7900 XT (gfx1100, 20 GB) unless
marked INFERRED. MEASURED means observed in a run; INFERRED means reasoned from measurements. The notes linked at
the end hold the detailed runs. Start with [README.md](README.md) for the short version.

## 1. Summary

- Goal: keep FP8 (E4M3) model weights as 8-bit bytes in VRAM on RDNA 3, which has no FP8 instructions, and widen
  them in registers, inside llama.cpp and PyTorch.
- llama.cpp, Llama 3.2 3B Instruct FP8, patched vs Q8_0 of the same weights (final build, FP8 file with a Q8_0 output
  layer):

| What | FP8 vs Q8_0 | Unmodified llama.cpp |
|---|---|---|
| Generation, one stream | +9 % (157.8 vs 145.0 t/s) | 66.2 t/s |
| 2-8 parallel streams | +3 % to +7 % | 2.2 to 2.9x slower than Q8_0 |
| Prompt processing, pp512 | +12 % to +17 % (8545-8962 vs 7660-7678 t/s) | NaN output |
| GPU memory, peak in a run | -4 % (4175 vs 4369 MiB) | |
| Answer quality | same as Q8_0, closer to exact | "GGGG..." on prompts over 8 tokens |

- Unmodified llama.cpp gives NaN for every FP8 model on RDNA 3 once more than 8 tokens are processed at once. The
  cause is found and the patch fixes it (section 4).
- PyTorch 2.13 (ROCm 7.1): every native FP8 matmul raises on gfx1100. A small extension that runs the decode kernel
  is 2.3 to 2.6x faster than PyTorch's own FP16 `F.linear` at half the weight memory (section 8).
- The dxil-spirv emulation of FSR 4's FP8 path on RDNA 3 does not round to the FP8 grid (section 10). An
  experimental patch exists; it is not yet tested in a game.
- All llama.cpp work is one patch, `bench/llama-cpp-fp8-mmvf-hip.patch` (about 470 lines, 4 files), for llama.cpp
  `6d3568e31`. Nothing has been sent upstream.

## 2. Setup

- llama.cpp `6d3568e31` (tag b12408), HIP build with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, ROCm/HIP 7.1.1 from
  the distribution, plus the `hipblas-devel` and `rocblas-devel` packages. A patched tree and an unpatched tree were
  built side by side.
- The test machine has a second ROCm device (the integrated GPU, gfx1036). llama.cpp and PyTorch both see it, and
  llama.cpp may put layers on it. All runs used `HIP_VISIBLE_DEVICES=0`, and the standalone programs refuse to run
  unless device 0 is gfx1100.
- Card limits, MEASURED (`bench/bandwidth.hip`, 1 GiB buffers, 16-byte loads): read 754 GB/s (median), copy 603 GB/s.
  Vendor spec is 800 GB/s.
- Out-of-memory warning: `llama-completion` without `-c` sizes the context to the free VRAM (89,344 tokens for the
  8B model, about 11 GB of KV cache). Always pass `-c`.

## 3. The core technique: E4M3 to FP16 by bit shifts

One 32-bit word of four E4M3 bytes b0..b3 becomes two FP16 pairs, 3 ALU operations per pair:

```
even = ((x << 8) & 0x80008000) | ((x << 7) & 0x3F803F80)   // halves (b0, b2)
odd  = ( x       & 0x80008000) | ((x >> 1) & 0x3F803F80)   // halves (b1, b3)
```

Each half equals the E4M3 value times 2^-8 exactly, subnormals included (an E4M3 subnormal m * 2^-9 lands on the
FP16 subnormal m * 2^-17). The factor 2^8 goes into the output scale. The pair order is k0, k2, k1, k3, so the
activations must be packed in the same order. A dot product does not care about the order.

- Checked against llama.cpp's own CPU conversion for all 2^32 words and every byte position: exact for all 254
  non-NaN codes, with no leak between bytes (`proto/widen_check.cpp`). A mutant with the odd mask widened to
  0x7F807F80 fails on 4,261,412,864 bytes. The 256-code table alone did not catch it, so test whole words.
- The NaN codes (0x7F, 0xFF) widen to +-480, not NaN. A per-word test, exact over all 2^32 words:
  `((x & 0x7F7F7F7F) + 0x01010101) & 0x80808080` is non-zero if and only if some byte is 0x7F or 0xFF. OR it into a
  per-row flag and write NaN rows at the end. This equals llama.cpp's behaviour because NaN times anything is NaN.
  Cost is about 0.75 ALU operations per weight, with no branch.
- With FP32 activations the widened half feeds `fmaf(float(half), a, acc)`, which compiles to `v_fma_mix_f32` (FP16
  times FP32 into FP32, one per weight per column, not dual-issuable). With FP16 activations, `v_dot2_f32_f16` does
  two weights per instruction.

## 4. Bug in unmodified llama.cpp: FP8 models give NaN on RDNA 3

- Symptom: perplexity is NaN for every chunk, and generation prints only "GGGG..." for any prompt longer than 8
  tokens.
- Cause: for more than 8 columns, `ggml_cuda_mul_mat_cublas` converts F8 weights to F16 and, on GPUs with fast F16,
  calls hipBLAS with `CUBLAS_COMPUTE_16F` and F16 output. Only Volta, RDNA 4 and CDNA get F32 output
  (`prefer_f32_output`). FP8 checkpoints store unscaled weights (up to 448) and apply the per-tensor scale (about
  1e-3 in the 3B model) in a later MUL, so the F16 output overflows 65504, becomes infinity, then NaN.
- Evidence (MEASURED, unpatched, 4 chunks, same file): CPU 9.7036; GPU with 8-token batches (F32 accumulation)
  9.7038; GPU with 512-token batches NaN; GPU with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` 9.7032. The weights have no
  NaN codes and the scales are finite.
- Why llama.cpp's tests miss it: their F8 test weights are uniform in [-1, 1]. Their MUL_MAT tolerance (NMSE 5e-4)
  also passes a 1/256 scale error; a 2x error is needed to fail.
- Fix 1, fallback, one line: F8 weights get F32 output in that function. Correct, but prompts drop to 1850 t/s on
  RDNA 3 (hipBLAS F32-output routines are slow here: 2306 us against 509 us at 8192 x 3072 x 512).
- Fix 2, scale folding: convert F8 to F16 with the per-tensor scale applied at llama.cpp's existing MUL_MAT + scale
  fusion hook, then use the normal fast F16 path. 6635 t/s and correct, but with F16 accumulation (KL divergence
  0.0011).
- Fix 3, final: a dedicated FP8 WMMA kernel at the same hook (section 6). Fixes 1 and 2 stay as fallbacks for shapes
  the kernel does not take.
- The same F16 overflow mechanism is behind open issue ggml-org/llama.cpp#30951 (other types, CUDA); master
  `9fba51d2` still had the code when checked. Not yet reported upstream for FP8 on RDNA 3.

## 5. Decode (1 to 8 tokens at once)

- Baseline cause (MEASURED, 16384 x 14336 weight): llama.cpp's FP8 decode loads 2 bytes per lane and converts each
  byte with `ldexpf` in software. At batch 1 it reaches 214 GB/s, 28 % of the 754 GB/s read ceiling. Q8_0 reaches
  721 GB/s and F16 739 GB/s. FP8 decode was 1.7x slower than F16, which reads twice the bytes.
- Batch 1, inside llama.cpp's `mul_mat_vec_f`: 16-byte loads per lane, shift widening, `fmaf` into FP32, NaN through
  the block sum. 312-322 us against 338-366 us for Q8_0 on the same weight.
- Batch 2 to 8, a separate kernel `mul_mat_vec_fp8_rows<ncols, rows_per_warp, ksplit>`:
  - Several rows per warp, so one activation load serves several rows. The original kernel reloads activations per
    row, which made n >= 4 slow.
  - `__builtin_nontemporal_load` on the weights (`global_load_b128 ... slc dlc`). Weights are read once and
    activations are reused. Without the hint the streaming weights evict the activations. This alone took n = 7 and 8
    from 1.04-1.10x to 0.93x Q8_0's time on the big weight.
  - `ksplit` = 2 warps per row group, splitting the row length (partial sums through LDS). Small model layers
    (1024 x 3072) need it: with 4 rows per warp only 64 blocks were left for 84 compute units.
  - Final setting, from sweeps in the models and on the big weight: 4 rows per warp and ksplit 2 for 2 to 8 columns;
    7 and 8 columns load the activations two columns at a time (below). FP8 then takes 0.74-0.92x Q8_0's time at every
    n = 1..8 on the 16384 x 14336 weight (notes/n8-gap-1.md).
- Did not help: `__launch_bounds__(128)` (no effect); converting each widened weight to FP32 so the products can
  dual-issue as `v_dual_fmac_f32` (the compiler pairs few; 2-3 %, within noise; 1.6-1.9x slower at 1 row per warp).
- Correctness, KL divergence against unmodified llama.cpp's exact F32 decode path (8-token batches as reference):
  0.000002-0.000004 at 1-, 3- and 8-token batches on 3B (same top token 99.9 %), 0.000002-0.000005 on 8B. Q8_0 is
  0.000605 (3B) and 0.000544 (8B) from the same reference.
- Eight parallel streams were the last gap (0.90x Q8_0 on 3B, 0.92x on 8B). The 8-column loop did 64-bit address math
  per column and waited on each column's loads; removing only the address math made the compiler hoist every load
  (about 190 registers, half the waves) and run slower. Loading the activations two columns at a time between
  scheduling barriers, from per-column pointers plus a 32-bit offset, with 4 rows per warp at 8 columns, fixed it:
  1.03x (3B) and 1.07x (8B) at 8 streams, and FP8 takes 0.74x Q8_0's time at n = 8 on the 16384 x 14336 weight
  (MEASURED, notes/n8-gap-1.md). 2 to 6 columns keep the old loop.

## 6. Prompt processing (more than 8 tokens): FP8 WMMA GEMM

- Kernel `mul_mat_fp8_wmma` in the patch (`fp8.cu`, RDNA 3 only through `#if defined(RDNA3)`):
  - tile 128 rows x 128 columns x 64 K, 8 waves (4 along rows, 2 along columns), one LDS buffer;
  - E4M3 widened to F16 while staging; F32 activations rounded to F16 while staging (no separate pass);
  - `v_wmma_f32_16x16x16_f16_w32` with F32 accumulation;
  - activations as the A operand and weights as B, so each accumulator lane runs along the weight rows and stores
    are contiguous;
  - columns past N are clamped on load; NaN rows as in decode; the scale is read from the scale tensor in the
    epilogue.
  It runs for more than 8 columns when M % 128 == 0, K % 64 == 0, and the tensors are contiguous and 16-byte aligned.
- How it got there: a first prototype (`proto/gemm`, v1) reached 37-41 TFLOPS on 16384 x 14336 and only 14-38
  TFLOPS on the 3B shapes. Register prefetch with double buffering (v2) gave 7-13 %. A configurable version (tile
  sizes, waves, split-K, fused activation conversion, rasterization, about 50 variants, with a hipBLAS baseline in the
  same harness) gave the best general configuration. Timed at N = 512 (us, hipBLAS F16 against ours without
  activation conversion): 8192x3072 512-534 vs 355-357; 3072x3072 156-157 vs 152-154; 1024x3072 60-62 vs 66-68;
  3072x8192 393-395 vs 388-391. The hipBLAS figures exclude llama.cpp's per-call weight conversion.
- In the model (3B, `llama-bench`): pp512 8545-8962 t/s against Q8_0 7660-7678 (+12 to +17 %); pp2048 with
  `-ub 2048` 9213-9225 against 7563-7657 (+20 %). On 8B, pp512 is 4048.5 against 3469.6 (+17 %).
- Prompt-path KL divergence against the exact reference: 0.000153 (3B) and 0.000172 (8B). Scale folding was
  0.001076, the F16 model on llama.cpp's own F16 path 0.002061, Q8_0 0.000605 (3B). Perplexity on 3B: FP8 9.4626,
  Q8_0 9.4743, F16 9.4942.
- Open: one tile configuration for all shapes; the small K/V layer is about 10 % behind hipBLAS in isolation.

## 7. End-to-end model tests

Llama 3.2 3B Instruct FP8 ([notes/model-test-1.md](notes/model-test-1.md); run 3B, final build):

- RedHatAI/Llama-3.2-3B-Instruct-FP8, compressed-tensors `float-quantized`, one scale per tensor, 4.4 GB. The
  converter keeps FP8 only for one-element scales; per-channel and 128-block checkpoints are converted to 16 bit.
- The checkpoint ships `lm_head` in BF16 (788 MB read per token). `proto/gguf_output_q8.py` makes a variant with a
  Q8_0 output layer and all other tensors byte-identical, so the comparison measures the FP8 weights.
- Real prompts, greedy ([notes/prompt-test-1.md](notes/prompt-test-1.md)): patched FP8 and Q8_0 replies are
  byte-identical and correct. Unmodified llama.cpp prints "GGGG...".
- Decode, `llama-batched-bench`, all sequences, t/s:

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 142.8 | 261.5 | 379.0 | 488.9 | 586.7 | 675.8 | 763.5 | 845.8 |
| FP8 patched | 156.6 | 280.8 | 405.0 | 524.7 | 621.3 | 706.0 | 771.2 | 758.6 |
| FP8 / Q8_0 | 1.10 | 1.07 | 1.07 | 1.07 | 1.06 | 1.04 | 1.01 | 0.90 |
| FP8 / Q8_0, final decode kernel (notes/n8-gap-1.md, 3 rounds) | 1.10 | 1.07 | 1.07 | 1.07 | 1.06 | 1.05 | 1.04 | 1.03 |
| FP8 unmodified | 65.7 | 113.8 | 161.1 | 201.9 | 241.4 | 273.7 | 271.5 | 297.1 |

  This table is from the run before the WMMA prompt kernel was added; decode kernels were not changed after it.
  Single stream, `llama-bench` tg128: Q8_0 145.0, FP8 (Q8_0 output) 157.8, FP8 as shipped 146.8, F16 94.5,
  unmodified FP8 66.2.

Llama 3.1 8B Instruct FP8 ([notes/model-test-8b.md](notes/model-test-8b.md); run `20261115-175739-llama31-8b`):

- RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8, `naive-quantized`, one BF16 scale per tensor. llama.cpp's converter asserts
  on this format although the weights and scales are the same shape; `bench/llama-cpp-convert-naive-fp8.patch` is a
  one-line fix.

| `llama-bench` | pp512 t/s | tg128 t/s |
|---|---|---|
| Q8_0, patched | 3469.6 | 76.3 |
| FP8 with Q8_0 output, patched | 4048.5 | 81.9 |
| FP8 as shipped, patched | 4085.6 | 78.0 |
| FP8 with Q8_0 output, unmodified | 2976.4 (NaN output) | 30.3 |

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 76.3 | 140.9 | 203.8 | 266.8 | 323.6 | 374.0 | 415.0 | 451.8 |
| FP8 patched | 81.7 | 149.2 | 218.5 | 279.9 | 340.8 | 395.8 | 417.5 | 417.5 |
| FP8 / Q8_0 | 1.07 | 1.06 | 1.07 | 1.05 | 1.05 | 1.06 | 1.01 | 0.92 |
| FP8 / Q8_0, final decode kernel (notes/n8-gap-1.md, 3 rounds) | 1.08 | 1.07 | 1.07 | 1.06 | 1.06 | 1.06 | 1.06 | 1.07 |

- Perplexity (50 chunks of a fixed local text): FP8 as shipped 7.2683, FP8 with Q8_0 output 7.2655, Q8_0 7.2763;
  unmodified FP8 NaN. Decode KL divergence 0.000002-0.000005; prompts 0.000172; Q8_0 0.000544.
- Real prompts: FP8 and Q8_0 differ in wording (both are proper answers). Q8_0 is a re-rounding of the FP8 weights, so
  greedy decoding can split at a near tie. The Python prime-check code was identical.

GPU memory (3B only, MEASURED): peak from the driver during a 4.5 KB prompt, minus the idle level before the run.

| Model | llama.cpp model buffer | Peak used by the run |
|---|---|---|
| FP8, Q8_0 output layer | 3088 MiB | 4175 MiB |
| Q8_0 | 3256 MiB | 4369 MiB |
| FP8 as shipped (BF16 output layer) | 3440 MiB | 4533 MiB |
| F16 | 6128 MiB | 7280 MiB |

FP8 saves 1/17 of the quantized weight bytes against Q8_0 (1 byte per weight instead of 1.0625). The prompt kernel
needs no temporary buffers (the scale-folding path allocated an F16 copy of each weight).

## 8. PyTorch on gfx1100

Notes: [pytorch-baseline-1](notes/pytorch-baseline-1.md), [pytorch-ext-1](notes/pytorch-ext-1.md).

- PyTorch 2.13.0+rocm7.1: `torch._scaled_mm` raises (only CUDA compute capability 9.0 or 8.9, or ROCm MI300 and up).
  Matmul and `F.linear` with an FP8 operand are not implemented. `.to()` conversions work. Workarounds at 16384 x
  14336: FP16 weights resident 826-945 us (569 GB/s); FP8 converted on every call 4057-4519 us.
- `proto/torch/fp8_linear.py`: `FP8Linear` keeps `float8_e4m3fn` weights in VRAM and calls the decode prototype
  through ctypes with PyTorch's stream. PyTorch ships its own `libamdhip64.so` with the same soname as the system
  one. Import torch first and both share PyTorch's copy (the script checks `/proc/self/maps`).
- 322-508 us (exact variant) and 330-403 us (FP16 activations) at n = 1..8, which is 2.3 to 2.6x faster than FP16
  `F.linear` and 8 to 14x faster than per-call conversion. Batch 1 to 8 only; no prefill, autograd or
  `torch.compile` integration.

## 9. Lessons for other RDNA 3 kernel work

- The 80 MB Infinity Cache fakes bandwidth. llama.cpp's own perf shape (4096 x 14336, about 60 MB as 8-bit) showed
  Q8_0 "reading" at 1737 GB/s. Use weights larger than about 4 x 80 MB, or rotate copies, for any memory-bound timing.
- LLVM splits 32-bit bit tricks into 16-bit operations when the result feeds an FP16 vector. An empty
  `asm("" : "+v"(x))` after the widening keeps the word whole (one variant went from 8.4 to 2.6 ALU ops per weight).
- `memcpy` into a `_Float16 __attribute__((ext_vector_type(16)))` from LDS words made LLVM move each half separately
  (784 VALU per 16 WMMA). `__builtin_bit_cast` of a `uint32 ext_vector_type(8), aligned(16)` load fixes it (52).
- `__builtin_nontemporal_load` needs a native vector type (`ext_vector_type`), not HIP's `uint4`.
- WMMA f32_16x16x16_f16_w32 layout, confirmed on the card: A lane%16 = row, B lane%16 = column, 16 halves of K per
  lane; D element e of lane t is row 2e + t/16, column t%16. Swapping operands (activations as A) makes the D lanes
  run along the weight rows, for contiguous stores.
- `v_fma_mix_f32` is what `fmaf((float) half, f32, f32)` compiles to, one per issue slot. Forcing FP32 conversion to
  get `v_dual_fmac_f32` did not pay.
- Small matrices leave the card idle. Count blocks against the 84 compute units (42 WGPs). Splitting K across warps
  (decode) or per-shape tiles (GEMM) is where the gains were.
- A loop-instruction counter on the `-save-temps` assembly (`proto/gemv/loop_isa.py`) caught every codegen problem
  above before any GPU time was spent. Pick loops by their math instruction, since a kernel can hold several loops.
- Tests must be able to fail. llama.cpp's NMSE 5e-4 gate passed a 1/256 scale error. Break the code on purpose and
  confirm red before trusting green. A per-output check (|error| / sum |w*a|) catches a single wrong row that NMSE
  hides.
- A kernel that looks instruction-bound may be memory-bound. At 11 ALU ops per weight the n = 8 decode kernel used
  about 40 % of the instruction rate (INFERRED from the instruction count, the measured time and the carry-over issue
  rate). The fix was a cache hint, not fewer instructions.
- Exact FP32 to E4M3 rounding is cheap: `carry-over/quant/e4m3_rdna3.h` does it in 7.7 cycles per value as FP16
  output and 11.3 as code bytes, bit-exact over all 2^32 inputs ([carry-over/quant/RESULTS.md](carry-over/quant/RESULTS.md)).

## 10. FSR 4's FP8 path on RDNA 3 (dxil-spirv)

### 10.1 What the emulation computes

Read in dxil-spirv `114e664` (pinned by vkd3d-proton `5b1089c8`), files `opcodes/dxil/dxil_ags.cpp` and
`spirv_module.cpp`:

- Without native FP8 (`wmma_fp8` off), FP8 matrices are held as F16 cooperative matrices.
- Where RDNA 4 converts an F32 accumulator to FP8 with saturation (`v_cvt_pk_fp8_f32`), the RDNA 3 workaround does an
  `OpFConvert` from F32 to F16 under a forced `RoundingModeRTZ` for 16-bit (set shader-wide for every WMMA shader on
  RDNA 3; the source comment says "FP16 RTZ allows faster conversions"), then `CoopMatSaturateFP8`, which only clamps
  to +-448. There is no rounding to the E4M3 grid. Values passed between FSR 4's layers keep F16 precision (11 bits)
  where RDNA 4 keeps 4. The RDNA 3 emulation therefore computes a different, more precise network than the one AMD
  ships for RDNA 4. It is not an approximation of it.
- FP8 loads (`build_packed_fp8_to_fp16_vec4`) already use the shift trick of section 3 (sign-extend, shift by 7, clear
  bit 14, multiply by 256 in F16). They are exact.
- FP8 stores to memory (`build_fp16_to_fp8`) round to nearest, ties toward zero, from F16. This is exact only for
  values already on the E4M3 grid.

### 10.2 The patch (not yet tested in a game)

`bench/dxil-spirv-rdna3-exact-fp8.patch`, on dxil-spirv `114e664`, 5 files, about 200 lines. It adds
`DXIL_SPIRV_CONFIG` values, each implying `wmma_rdna3_workaround`:

- `wmma_rdna3_exact_fp8`: where an F32 source becomes FP8 with saturation, a new `CoopMatRoundFP8F32` helper first
  rounds each element to the E4M3 grid in F32 (the carry-over magic-constant method: C = 1.5 * 2^(E+20) from
  clamp(|x|, 2^-6, 448), then (|x| + C) - C with `NoContraction` on both operations, clamp to [0, 448], NaN kept, sign
  restored). The F16 conversion is then exact. F16 sources use `CoopMatRoundFP8F16` instead of the clamp-only helper.
- `wmma_rdna3_exact_fp8_rte`: the same, without forcing RTZ for F16, so the rest of FSR 4's F16 math rounds as on
  RDNA 4.
- With the option unset, or set to `wmma_rdna3_workaround`, the output is unchanged.

Checks, MEASURED, with dxil-spirv's own `test_shaders.py --subfolder ags` (DXC v1.10.2605.24, scratch copies):

- With the option off, all 30 AGS shaders match the stored references.
- All 30 validate in the workaround, exact and exact_rte modes.
- Only `cs_wmma_fp32_fp8_conversions.sm66.ssbo.nv-coopmat2` differs between the workaround and exact modes. That is the
  `--full-wmma 0 1` route, which should be the RDNA 3 + RADV route; this is to be confirmed in a game.
- A first version produced invalid SPIR-V (an `OpSelect` used an `OpIsNan` emitted after it). It was fixed and
  revalidated. Constants and decorations were checked in the disassembly.

Expected effect (INFERRED): FSR 4 output much closer to RDNA 4's, but not bit-identical (RDNA 3's F16 WMMA sums mixed
signs slightly differently, see [carry-over/measurements/](carry-over/measurements/matrix-arithmetic.md)), and
probably a little slower. No in-game run has been done. `bench/fsr4-game.sh snapshot | use base|exact | restore |
status` switches the vkd3d-proton and dxil-spirv builds in a game folder and restores it with a full hash check; one
round trip with the game closed restored all 139 files.

## 11. Open items

- Prompt kernel: per-shape tile choice or split-K for the small K/V layers.
- No independent review of the llama.cpp patch yet.
- Untested: other models and sizes, other RDNA 3 cards, RDNA 4 (the WMMA kernel is RDNA 3 only by design; RDNA 4
  has native FP8 and a different WMMA layout), MUL_MAT_ID (mixture-of-experts) shapes, Vulkan.
- The bug report for llama.cpp is not filed yet.

## 12. Files and notes

- Patches: `bench/llama-cpp-fp8-mmvf-hip.patch` (apply to llama.cpp `6d3568e31` with `git apply`),
  `bench/llama-cpp-convert-naive-fp8.patch`, `bench/llama-cpp-out-of-cache-shape.patch` (benchmark only),
  `bench/dxil-spirv-rdna3-exact-fp8.patch`.
- Scripts: `bench/run-gpu.sh`, `bench/run-model.sh`, `bench/fsr4-game.sh`, `bench/bandwidth.hip`.
- Prototypes: `proto/gemv/fp8_gemv.hip`, `proto/gemm/fp8_gemm.hip` (link with `-lhipblas`), `proto/widen_check.cpp`,
  `proto/torch/`.
- Notes: [survey](notes/survey.md), [go-no-go-1](notes/go-no-go-1.md),
  [prototype-results-1](notes/prototype-results-1.md), [llama-cpp-integration-1](notes/llama-cpp-integration-1.md),
  [model-test-1](notes/model-test-1.md), [model-test-8b](notes/model-test-8b.md),
  [prompt-test-1](notes/prompt-test-1.md), [pytorch-baseline-1](notes/pytorch-baseline-1.md),
  [pytorch-ext-1](notes/pytorch-ext-1.md).
