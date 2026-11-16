# FP8 on AMD RDNA 3: limits and workarounds

A write-up of what FP8 (E4M3) looks like on Radeon RX 7000 GPUs today, which parts are hard and why, and the
workarounds I found. Tested on an RX 7900 XT (gfx1100, 20 GB).

RDNA 3 has no FP8 instructions. Most software therefore treats FP8 as a second-class format on these cards: slow,
sometimes wrong, or refused. I wanted to know how much of that is the hardware and how much is software that was never
tuned for it. For llama.cpp the answer is: mostly software. With the changes described here, FP8 models in llama.cpp on
an RX 7900 XT give correct answers and run faster than the 8-bit Q8_0 format: 7 to 9 % faster for one stream, 3 to
7 % for two to eight parallel streams (the smallest gains are close to the run-to-run spread), 12 to 20 % faster prompt
processing, with about 5 % less memory. Before the changes, FP8 was about 2 to 3 times slower
than Q8_0, and any prompt longer than 8 tokens gave NaN (the model printed "GGGG...").

Status: research on one GPU and two models. Nothing has been sent to llama.cpp, PyTorch or dxil-spirv upstream yet.

## Contents

1. [Terms](#terms)
2. [The state of FP8 on RDNA 3](#the-state-of-fp8-on-rdna-3)
3. [Architecture limitations](#architecture-limitations)
4. [Workarounds](#workarounds)
5. [Results](#results)
6. [Applying it](#applying-it)
7. [Limits](#limits)
8. [How this was made](#how-this-was-made)
9. [Licence and credits](#licence-and-credits)

## Terms

- **FP8 (E4M3):** an 8-bit float with 4 exponent bits and 3 mantissa bits, largest value 448. FP8 model files store
  weights in it, plus a scale per tensor that the weights are multiplied by.
- **Q8_0:** llama.cpp's own 8-bit format, the usual comparison point for an 8-bit model.
- **Decode / prompt processing:** producing one token per step (1 to 8 tokens at once with parallel streams) versus
  reading the prompt in large batches. Speeds are in tokens per second (t/s).
- **WMMA:** RDNA 3's matrix instruction (16 x 16 x 16 tiles, FP16 or 8-bit integer inputs).
- **Wave (warp):** a group of 32 GPU threads that run together. llama.cpp's code calls it a warp. **LDS:** the fast
  memory shared by the threads of one block.
- **Perplexity, KL divergence:** measures of how much a model's predictions change. Lower perplexity is better; a KL
  divergence near 0 means the outputs match a reference almost exactly.
- **FSR 4:** AMD's AI upscaler for games, whose neural network uses FP8. On Linux it reaches RDNA 3 through
  vkd3d-proton (DirectX 12 on Vulkan) and its shader translator dxil-spirv, which emulate the FP8 instructions.

## The state of FP8 on RDNA 3

I list each problem with the project that owns it. These are descriptions of the current state, not complaints. Most of
the software involved is built for newer hardware first, and FP8 on RDNA 3 is a corner case for all of it.

| Problem | Owner | What I saw |
|---|---|---|
| No FP8 instructions in the vector units or in WMMA | AMD (hardware) | AMD's precision table lists FP8 as supported on RDNA 4 and CDNA, not on RDNA 3. Nothing to fix here, only to work around. |
| FSR 4 is officially RDNA 4 only | AMD | As far as I know, FSR 4 runs on RDNA 3 only through community translation layers (vkd3d-proton and dxil-spirv). |
| FP8 decode is slow | llama.cpp | The HIP path converts one byte at a time in software. At batch 1 it reads 214 GB/s of weights, 28 % of the 754 GB/s the card can read. Q8_0 reaches 721 GB/s, F16 739 GB/s. |
| FP8 models give NaN on prompts | llama.cpp | The default is wrong for FP8 on RDNA 3. For more than 8 tokens at once, llama.cpp converts FP8 weights to FP16 and asks hipBLAS for FP16 output, and the result overflows (INFERRED; forcing FP32 output removes the NaN). llama.cpp's own setting `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` already avoids it, but prompts then run at a quarter to a third of Q8_0's speed. Still the case on master `f477c1b90` (2026-11-16). Section [Avoiding the FP16 overflow](#e-avoiding-the-fp16-overflow). |
| FP8 matmul is refused | PyTorch | `torch._scaled_mm` and FP8 `F.linear` raise on gfx1100. The error names only MI300 and newer for ROCm; PyTorch's architecture list (read, not run) also has RDNA 4. |
| FP8 only for some checkpoints | llama.cpp | The converter keeps FP8 only when each tensor has one scale. Per-channel and 128-block checkpoints are converted to 16 bit. It also asserts on the compressed-tensors `naive-quantized` variant, which has the same layout. |
| FSR 4's FP8 path keeps FP16 precision | dxil-spirv (community) | From reading the code: the RDNA 3 emulation converts FP32 to FP16 and clamps to +-448. It never rounds to the FP8 grid. |

Five of these have a fix or partial fix in this repo: decode speed and the NaN bug (one llama.cpp patch), the
`naive-quantized` checkpoints (a one-line converter patch; per-channel and block scales are still converted to 16 bit),
FP8 in PyTorch (a small extension), and the FSR 4 rounding (a dxil-spirv patch, not yet tested in a game). The hardware
limits stay.

## Architecture limitations

MEASURED means I (or the agents under my direction) saw it in a run on the 7900 XT. INFERRED means reasoned from
measurements. Full tables are in [measurements/](measurements/).

### No FP8 instructions

RDNA 3 has no FP8 instruction in the vector ALU (VALU) and none in WMMA, the matrix instruction. The Vulkan driver
reports 14 cooperative matrix types for the card: integer 8-bit, FP16 and mixed forms. No BF16 and no FP8 entries
(MEASURED, [measurements/matrix-arithmetic.md](measurements/matrix-arithmetic.md)). An FP8 weight has to be widened to
FP16 by ordinary instructions before any math happens.

### What RDNA 3 has instead

Measured on the 7900 XT, one SIMD, for 4096 products (one 16x16x16 block), from
[measurements/shader-throughput.md](measurements/shader-throughput.md):

| Path | Instructions | Cycles per 4096 products |
|---|---|---|
| FP32 FMA | 128 `v_fmac_f32` | 137 |
| FP16 pairs, FP32 accumulate | 128 `v_fma_mix_f32` | 132 |
| Packed FP16 FMA, FP16 accumulate | 64 `v_pk_fmac_f16` | 75 |
| Packed int8 dot | 32 `v_dot4_i32_iu8` | 63 |
| Matrix, FP16 inputs | 1 `v_wmma_f32_16x16x16_f16` | about 33 |
| Matrix, int8 inputs | 1 `v_wmma_i32_16x16x16_iu8` | about 32 |

What this means for FP8 weights:

- WMMA is about 2x faster than the best vector path and about 4x faster than FP32 or FP16-mix FMAs. For prompts,
  where there are many columns to amortise the widening, WMMA with FP16 inputs is the right target.
- The int8 dot runs at half rate (INFERRED: 32 instructions in 62.6 cycles).
- `v_fma_mix_f32` (FP16 times FP32 into FP32) issues one per slot, and the compiler does not pair it into a dual-issue
  (VOPD) instruction. A weight widened to FP16 costs one `v_fma_mix_f32` per weight per column. `v_dot2_f32_f16` does
  two weights per instruction, but it needs FP16 activations.
- WMMA and vector work share the SIMD and do not overlap (MEASURED). Four F16 WMMAs plus 128 FMAs took 275.7 cycles;
  the parts alone take 258.9 in total. So widening code is added on top of the matrix time, never hidden under it.
- Widening an E4M3 byte to FP16 with bit operations costs about 35 cycles per 8 codes (40 when exact), measured in the
  Vulkan probe.

### FP16 has a smaller range than FP8 weights need

E4M3 weights go up to 448. FP8 checkpoints store them unscaled and apply a per-tensor scale (about 1e-3 in the 3B model)
later. FP16 tops out at 65504. A dot product over thousands of unscaled weights goes past that easily. If accumulation or
output is FP16, the result becomes infinity and then NaN. This is the cause of the llama.cpp NaN bug (INFERRED: the
infinity itself was not observed, but forcing FP32 output removes the NaN, see section e), and it limits any FP16 route
on RDNA 3.

### The 80 MB Infinity Cache hides memory limits

llama.cpp's own perf shape (4096 x 14336, about 60 MB as 8-bit) fits in the cache. It made Q8_0 look like it was reading
at 1737 GB/s. For memory-bound timing, use weights larger than about 4 x 80 MB or rotate copies. I used 16384 x 14336
(235 MB as FP8).

### Bandwidth and size of the card

- Read ceiling 754 GB/s (median, 1 GiB buffers, 16-byte loads). Copy 603 GB/s. The vendor figure is 800 GB/s
  ([examples/bandwidth/bandwidth.hip](examples/bandwidth/bandwidth.hip)).
- 84 compute units (42 WGPs). Small layers leave most of the card idle. In the 3B model the K and V matrices are
  1024 x 3072. With 4 rows per warp the decode kernel had only 64 blocks for 84 compute units. At 512 prompt columns
  that layer is only 32 tiles of 128 x 128.

### F16 WMMA sums are slightly inexact

When the terms have mixed signs, `v_wmma_f32_16x16x16_f16` is off by a few units of 2^(E-24), where E is the exponent of
the largest term. Even 1 + (-1) returns -2^-24, not 0 (MEASURED). The result depends on the order of the K values, and
none of the 13 arithmetic models I tried fits it. It is deterministic. The effect is small:

- On random E4M3 data, 2 per million outputs change their E4M3 code after rounding. On one sparse test set,
  451 per million.
- So FP8 results from RDNA 3's FP16 WMMA cannot be bit-identical to RDNA 4's FP8 WMMA. They are close.

Details: [measurements/matrix-arithmetic.md](measurements/matrix-arithmetic.md) and
[measurements/errors-after-quantizing.md](measurements/errors-after-quantizing.md).

### Compiler pitfalls on gfx11

All three cost me real time. Each was caught by counting instructions in the assembly, not by timing
([examples/tools/loop_isa.py](examples/tools/loop_isa.py)).

- LLVM splits 32-bit bit tricks into 16-bit operations when the result feeds an FP16 vector. One variant took 8.4 ALU
  operations per weight instead of 2.6. An empty `asm("" : "+v"(x))` after the widening keeps the word whole.
- `memcpy` from LDS words into an FP16 vector made LLVM move each half separately: 784 vector instructions per 16 WMMAs.
  `__builtin_bit_cast` of a 32-bit vector load brought it to 52.
- The compiler pairs few FMAs into dual-issue (VOPD) instructions, because the operands must sit in different register
  banks and HIP gives no control over that. Also `__builtin_nontemporal_load` needs a native `ext_vector_type`, not HIP's
  `uint4`.

### Findings beyond FP8 arithmetic: kernels and drivers

A second workload on the same card (the convolution and transformer kernels of an image-upscaling network, in HIP and
in D3D12 under Proton) turned up findings that are not about FP8 rounding. Details, research steps and sources are in
[measurements/kernels-and-drivers.md](measurements/kernels-and-drivers.md).

- On gfx11 the compiler can move a `v_wmma` chain under a per-lane mask. The result is NaN or wrong values on partly valid
  tiles, which a CPU emulator cannot show. Fixed in the source, with a scan of the disassembly that finds every such site.
- `v_permlanex16` with the fetch-inactive bit set reads the register of a switched-off lane (MEASURED, five mask shapes).
- F16 matrix ops through D3D12 under Proton (vkd3d-proton, dxil-spirv, RADV) equal the HIP instruction bit for bit. The
  raw fragment needs a replicated lane layout and covers only K = 0..7 per op.
- Headless Wine needs its null graphics driver and a scratch `XDG_RUNTIME_DIR`. Under Wine and Proton the
  physical-adapter-ids query of `D3DKMTQueryAdapterInfo` is not implemented.
- Unchanged RDNA 4 FP8 shaders run on RDNA 3 when dxil-spirv emulates FP8 on raw fragments. About 70 % of the cost is FP8
  decode; a prototype with shift widening and one full-size matrix op cut a synthetic whole pass from about 32 ms to
  about 11 ms.

## Workarounds

### a. E4M3 to FP16 by bit shifts

One 32-bit word holds four E4M3 bytes b0..b3. Two FP16 pairs come out in 3 operations per pair:

```
even = ((x << 8) & 0x80008000) | ((x << 7) & 0x3F803F80)   // halves (b0, b2)
odd  = ( x       & 0x80008000) | ((x >> 1) & 0x3F803F80)   // halves (b1, b3)
```

Each half is the E4M3 value times 2^-8, exactly, subnormals included. The factor 2^8 goes into the output scale, so no
multiply is needed per weight. The pair order is k0, k2, k1, k3, so the activations must be packed in the same order. A
dot product does not care about the order.

Why it works here: it needs only shifts and masks, no table and no float math. The same trick shows up in other
projects (for example FP8 Marlin on NVIDIA); the new part is the exactness check below.

Checks (MEASURED): all 2^32 words and every byte position match llama.cpp's own CPU conversion for all 254 non-NaN
codes, with no leak between bytes ([examples/widen-check/widen_check.cpp](examples/widen-check/widen_check.cpp)). A
deliberately broken mask (odd mask widened to 0x7F807F80) fails on 4,261,412,864 bytes. The 256-code table alone did not
catch it, so test whole words.

NaN codes (0x7F and 0xFF) widen to +-480, not NaN. llama.cpp treats them as NaN, so I check each word:
`((x & 0x7F7F7F7F) + 0x01010101) & 0x80808080` is non-zero exactly when some byte is a NaN code. The kernel ORs that
into a per-row flag and writes NaN for those rows at the end. This equals llama.cpp's behaviour. It costs about 0.75 ALU
operations per weight and no branch.

### b. Exact FP32 to E4M3 rounding

Writing FP8 (activations, KV cache, FSR 4 layer outputs) needs FP32 to E4M3 with round-to-nearest-even and saturation at
448. A software version does about 16 operations per value. A faster exact method:

1. Clamp |x| to [2^-6, 448] with one `v_med3_f32`. Take its exponent E.
2. Build the constant C = 1.5 * 2^(E+20) from the exponent bits.
3. Compute `y = |x| + C` then `r = y - C`. The FP32 add rounds |x| to a multiple of 2^(E-3), nearest even, because C's
   mantissa is even. The subtract is exact. The floor of 2^-6 gives the E4M3 subnormal step of 2^-9.
4. Clamp to 448, keep NaN, restore the sign.

| Output | Baseline | This method |
|---|---|---|
| FP16 bits, ready as the next matrix operand | 17.9 cycles per value | 7.7 cycles per value |
| E4M3 code bytes | 17.9 cycles per value | 11.3 cycles per value |

Both are bit-exact over all 2^32 FP32 inputs, NaN and denormals included (MEASURED). Two plausible-looking variants
failed the same check and were rejected: a scaled constant without the exponent mask (224 wrong inputs) and a Veltkamp
split (ties wrong on 245 million inputs). The code is [examples/quant/e4m3_rdna3.h](examples/quant/e4m3_rdna3.h);
results in [measurements/e4m3-rounding.md](measurements/e4m3-rounding.md). This method is also what the dxil-spirv
patch uses (section g).

### c. Decode kernels (1 to 8 tokens at once)

Decode is memory-bound: one pass over the weights per token. The goal is to read FP8 bytes at close to the card's read
speed and spend few instructions per weight.

- Wide loads. Each lane loads 16 bytes of weights, widens them with the shifts above, and accumulates
  `fmaf(float(half), a, acc)` against FP32 activations. That compiles to `v_fma_mix_f32`. FP32 activations keep the
  products exactly as llama.cpp computes them. At batch 1 this reaches 741 GB/s, 98 % of the read ceiling, and takes
  312-322 us against 338-366 us for Q8_0 on the same weight.
- Several rows per warp. One activation load then serves several rows. The original kernel reloaded activations per row,
  which made 4 or more columns slow.
- Non-temporal weight loads (`global_load_b128 ... slc dlc`). Weights are read once and activations are reused.
  Without the hint, the streaming weights evict the activations from cache. This alone moved 7 and 8 columns from
  1.04-1.10x to 0.93x Q8_0's time. At 11 vector operations per weight the kernel had looked instruction-bound, but it used only
  about 40 % of the instruction rate (INFERRED). It was waiting on memory, and the fix was a cache hint, not fewer
  instructions.
- Two warps per row for small layers (`ksplit` 2, partial sums through LDS). The 1024 x 3072 layers needed this to
  keep the 84 compute units busy.
- Eight streams, the last gap. The 8-column loop did 64-bit address math per column and waited on each column's loads.
  Removing only the address math made the compiler hoist every load (about 190 registers, half the waves) and run
  slower. What worked was loading the activations two columns at a time, between scheduling barriers, from per-column
  pointers plus a 32-bit offset, with 4 rows per warp. Registers decide speed here: kernels at or below about 64
  registers were fast, those above 100 were 10 to 30 % slower.

Effect on the 16384 x 14336 weight: FP8 takes 0.74-0.92x of Q8_0's time at every batch size from 1 to 8. The prototype is
[examples/decode-gemv/fp8_gemv.hip](examples/decode-gemv/fp8_gemv.hip); the production version is in the llama.cpp patch.

### d. Prompt processing: a WMMA kernel that widens while staging

For more than 8 tokens, llama.cpp converts the whole weight to FP16 on every call and then runs hipBLAS. My kernel
skips the conversion pass:

- Tile of 128 rows x 128 columns x 64 K, 8 waves, one LDS buffer.
- E4M3 is widened to FP16 while the weights are copied into LDS, so the widening is paid once per tile.
  FP32 activations are rounded to FP16 in the same step.
- `v_wmma_f32_16x16x16_f16_w32` with FP32 accumulation, so the sums cannot overflow FP16.
- Activations are the A operand and weights the B operand. Then each accumulator lane runs along the weight rows and
  stores are contiguous.
- The per-tensor scale is applied in the epilogue. NaN rows are written as in decode.

Timed in isolation at 512 columns (microseconds, lower is better; the hipBLAS figures exclude llama.cpp's per-call
weight conversion):

| Layer shape | hipBLAS FP16 | This kernel |
|---|---|---|
| 8192 x 3072 | 512-534 | 355-357 |
| 3072 x 3072 | 156-157 | 152-154 |
| 3072 x 8192 | 393-395 | 388-391 |
| 1024 x 3072 | 60-62 | 66-68 |

The small K/V shape is about 10 % behind. The prototype went through register prefetch with double buffering (7 to 13 %
faster) and then a configurable version with about 50 tile variants and a hipBLAS baseline in the same harness.
Prototype: [examples/prompt-gemm/fp8_gemm.hip](examples/prompt-gemm/fp8_gemm.hip).

### e. Avoiding the FP16 overflow

Evidence for the NaN bug, with the same file and text and 4 chunks of perplexity, unmodified llama.cpp:

| Run | Perplexity |
|---|---|
| CPU | 9.7036 |
| GPU, 8-token batches (F32 accumulation) | 9.7038 |
| GPU, 512-token batches | NaN |
| GPU, 512-token batches with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 9.7032 |

The weights have no NaN codes and the scales are finite. llama.cpp's tests miss it because their F8 test weights are
uniform in [-1, 1], and the MUL_MAT tolerance (NMSE 5e-4) passes even a 1/256 scale error. I found that by changing the
scale in my kernel on purpose.

**Retest on current llama.cpp (MEASURED, master `f477c1b90`, 2026-11-16, unmodified, 4 chunks).** The bug is still
there, and llama.cpp's existing override already avoids it. So the default is wrong for FP8 on RDNA 3; nothing is
missing.

| Setting | Llama 3.2 3B FP8 | Llama 3.1 8B FP8 |
|---|---|---|
| Default | NaN | NaN |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` (older name, removed from llama.cpp in July 2026) | NaN | NaN |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 9.7032 | 7.4408 |
| 8-token batches (reference) | 9.7038 | 7.4407 |

The older setting name is no longer read and gives no warning, so advice that names it no longer helps. The current
override is correct but slow on prompts: 1906 t/s at 512 tokens on 3B and 1329 t/s on 8B, against 7332 and 3543 for
Q8_0 on the same build. It does not change generation, which stays at 66.7 t/s on 3B (Q8_0: 148.3) because the FP8
decode path is slow regardless.

Three fixes, in the patch, in order of preference:

1. The WMMA kernel above (FP32 accumulation, scale in the epilogue). Prompt speed on 3B: 8545-8962 t/s.
2. Fold the per-tensor scale into the FP16 conversion, then use llama.cpp's normal FP16 path. The weights become small
   enough that nothing overflows. 6635 t/s and correct, but with FP16 accumulation (KL divergence 0.0011).
3. Last resort, FP32 output for FP8 weights. It is correct but slow on RDNA 3: 1850 t/s. hipBLAS's FP32-output routines
   take 2306 us where the FP16-output ones take 509 us (8192 x 3072 x 512).

Fixes 2 and 3 stay as fallbacks for shapes the kernel does not take.

### f. FP8 in PyTorch without native support

`torch._scaled_mm` raises on gfx1100, and `F.linear` with an FP8 weight is not implemented. Converting FP8 to FP16 on
every call works but costs 4057-4519 us at 16384 x 14336, against 826-945 us for FP16 weights kept resident (first
measurement run).

`FP8Linear` keeps the weight as `float8_e4m3fn` in VRAM and calls the decode kernel through ctypes on PyTorch's stream.
At 16384 x 14336 and batch 1 to 8 (second run, so the FP16 times differ slightly from the first):

| Path | Time |
|---|---|
| PyTorch `F.linear`, FP16 weights (2x the memory) | 820-947 us |
| `FP8Linear`, exact variant | 322-508 us |
| `FP8Linear`, FP16 activations | 330-403 us |
| FP8 converted on every call | 4057-4519 us |

Against PyTorch's FP16 path that is 1.9 to 2.6x faster for the exact variant and 2.3 to 2.6x for the FP16-activation
variant, at half the weight memory. It handles batch 1 to 8 only: no prompt
path, no autograd, no `torch.compile`. PyTorch ships its own `libamdhip64.so` with the same name as the system one.
Import torch first and both share PyTorch's copy. Files: [examples/pytorch/](examples/pytorch/).

### g. FSR 4: exact FP8 rounding for dxil-spirv (NOT YET TESTED IN A GAME)

FSR 4 can run on RDNA 3 through vkd3d-proton and dxil-spirv. I read dxil-spirv `114e664` (pinned by vkd3d-proton
`5b1089c8`). Without native FP8, it holds FP8 matrices as FP16 cooperative matrices. Where RDNA 4 converts an FP32
accumulator to FP8 with saturation, the RDNA 3 path converts FP32 to FP16 with round-toward-zero and then clamps to
+-448. It never rounds to the E4M3 grid. So values passed between FSR 4's layers would keep 11 bits of precision where
RDNA 4 keeps 4, and the RDNA 3 emulation would compute a different, more precise network than the one AMD ships for RDNA
4, not an approximation of it (INFERRED from the code; FSR 4 itself has not been run). (FP8 loads already use the shift
trick and are exact. Stores round to nearest, ties toward zero, from FP16, which is exact only for values already on the
grid.)

The patch adds two `DXIL_SPIRV_CONFIG` options that round to the E4M3 grid first, using the magic-constant method from
section b:

- `wmma_rdna3_exact_fp8`: round FP32 to the grid before the FP16 conversion, and FP16 sources through an exact helper.
- `wmma_rdna3_exact_fp8_rte`: the same, without forcing round-toward-zero, so the rest of FSR 4's FP16 math rounds as on
  RDNA 4.

With both unset, the output is unchanged. What I checked: all 30 AGS test shaders in dxil-spirv's own suite match their
references with the option off and validate in all three modes. Only one shader differs between the workaround and the
exact modes, on the route that should be the one RDNA 3 with RADV takes; that is still to be confirmed in a game. A
first version produced invalid SPIR-V (a select used a NaN test emitted after it); it was fixed and revalidated.

What I have not done: **run it in a game.** I do not know yet whether the image gets closer to RDNA 4's, or how much
slower it is. My expectation (INFERRED) is closer but not bit-identical, because of the WMMA inexactness above, and a bit
slower. Treat the patch as an experiment.

### h. What did not help

| Idea | Result |
|---|---|
| `__launch_bounds__(128)` on the decode kernel | No effect |
| Convert each widened weight to FP32 so FMAs can dual-issue (`v_dual_fmac_f32`) | 2-3 %, within noise; 1.6-1.9x slower at 1 row per warp |
| 8 warps per block at 8 columns | 411 t/s, no change from the 407-416 baseline |
| 17 combinations of rows per warp and `ksplit` at 8 columns | 207-411 t/s, none better |
| Removing address math only, letting the compiler hoist loads | 284 t/s against 410, about 190 registers |
| Capping registers with `amdgpu_waves_per_eu(16)` | Spills: 96 registers plus 68 scratch |
| Scheduling-barrier groups for 2 to 6 columns | Slower (2 columns: 128 against 148 t/s); kept the old loop |
| FP16 activations as the default | Faster for 4 or more columns, but rounds the activations and changes results slightly. Kept as an option only |
| A lookup table in LDS for widening | Slower than the bit form in the Vulkan probe |

## Results

All numbers are from an RX 7900 XT, ROCm/HIP 7.1.1, llama.cpp `6d3568e31` with and without
[patches/llama.cpp-fp8-rdna3.patch](patches/llama.cpp-fp8-rdna3.patch). Tokens per second (t/s), higher is better.
"Q8_0" is the same weights re-rounded to llama.cpp's 8-bit format. The FP8 file used for the comparison has its output
layer converted to Q8_0, so the comparison measures the FP8 weights and not a 16-bit output layer. The checkpoints ship
that layer in BF16, and the numbers for the files as shipped are in the notes below.

### Llama 3.2 3B Instruct FP8

| What | Q8_0 | FP8, patched | FP8 vs Q8_0 | FP8, unmodified llama.cpp |
|---|---|---|---|---|
| One stream, generation | 145.0 | 157.8 | +9 % | 66.2 |
| 2-8 parallel streams (ratio of decode speed) | | | +3 % to +7 % | 2.2 to 2.9x slower than Q8_0 |
| Prompt, 512 tokens | 7660-7678 | 8545-8962 | +12 % to +17 % | NaN |
| Prompt, 2048 tokens (`-ub 2048`) | 7563-7657 | 9213-9225 | +20 % | NaN |
| Model buffer in GPU memory | 3256 MiB | 3088 MiB | -5 % | |
| Peak GPU memory in a run | 4369 MiB | 4175 MiB | -4 % | |
| Perplexity (lower is better) | 9.4743 | 9.4626 | same or slightly better | NaN |

Notes:

- The 2-8 stream ratios are from the final decode kernel, medians of 3 rounds of `llama-batched-bench`. FP8 over Q8_0
  for 1 to 8 streams: 1.10, 1.07, 1.07, 1.07, 1.06, 1.05, 1.04, 1.03. The machine ran a little slower during these runs
  (Q8_0 itself was 2-3 % lower), so compare ratios, not absolute speeds.
- FP8 as shipped (BF16 output layer) is 146.8 t/s at one stream and 3440 MiB / 4533 MiB for memory. F16 is 94.5 t/s,
  6128 MiB / 7280 MiB, perplexity 9.4942.
- Two real prompts (a refrigerator explanation and a Python prime check): patched FP8 and Q8_0 replies were identical
  word for word. Unmodified llama.cpp printed only "GGGG...".

### Llama 3.1 8B Instruct FP8

| What | Q8_0 | FP8, patched | FP8 vs Q8_0 | FP8, unmodified llama.cpp |
|---|---|---|---|---|
| One stream, generation | 76.3 | 81.9 | +7 % | 30.3 |
| 2-8 parallel streams (ratio of decode speed) | | | +6 % to +7 % | 2.7 to 3.3x slower than Q8_0 |
| Prompt, 512 tokens | 3469.6 | 4048.5 | +17 % | NaN |
| Perplexity (50 chunks, lower is better) | 7.2763 | 7.2655 | same or slightly better | NaN |

FP8 over Q8_0 for 1 to 8 streams: 1.08, 1.07, 1.07, 1.06, 1.06, 1.06, 1.06, 1.07. GPU memory was not measured for this
model. With greedy decoding the FP8 and Q8_0 replies differ in wording on this model (both are proper answers). Q8_0 is a
re-rounding of the FP8 weights, so a near tie between two words can go either way.

### Closeness to llama.cpp's exact path

KL divergence of the patched build against unmodified llama.cpp's own FP32 decode path (8-token batches as reference);
lower is closer.

| | 3B | 8B |
|---|---|---|
| Decode kernels, 1, 3 and 8 tokens | 0.000002-0.000004 | 0.000002-0.000005 |
| Prompt kernel, 512 tokens | 0.000153 | 0.000172 |
| Q8_0 for scale | 0.000605 | 0.000544 |
| Scale folding (fix 2 above) | 0.001076 | |
| F16 model on llama.cpp's F16 path | 0.002061 | |

The top token is the same in 99.9 % of positions for decode.

## Applying it

The patches are written against llama.cpp commit `6d3568e31` (tag b12408). They also apply to master `f477c1b90`
(2026-11-16); on that build the FP8 tests pass (448 of 448) and, against Q8_0, FP8 generation is 10 % faster on 3B and
8 % on 8B, and 512-token prompts 12 % and 17 % faster (MEASURED, `llama-bench`, 3 repetitions). On other commits they
may not apply.

Without patching: set `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` to get correct output from FP8 models on RDNA 3. Prompts are
slow with it (see section e), and generation stays at the unpatched speed.

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 6d3568e31
git apply /path/to/rdna3-fp8/patches/llama.cpp-fp8-rdna3.patch
# only for compressed-tensors "naive-quantized" checkpoints (for example the 8B model):
git apply /path/to/rdna3-fp8/patches/llama.cpp-convert-naive-fp8.patch
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- The build needs the ROCm development packages, including hipBLAS and rocBLAS headers (`hipblas-devel` and
  `rocblas-devel` on Fedora).
- The patch adds about 500 lines in 4 files. The prompt kernel runs only on RDNA 3. Other GPUs keep the old behaviour.
- On the final build, `test-backend-ops test -b ROCm0 -p f8_e4m3` passes 450 of 450 cases. I broke the scale in the
  kernel on purpose and the MUL_MAT cases failed, then restored it.
- **Which checkpoints stay FP8.** llama.cpp's converter keeps FP8 weights only when each tensor has one scale. That is
  the case for `RedHatAI/Llama-3.2-3B-Instruct-FP8` (`float-quantized`) and, with the second patch, for
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` (`naive-quantized`). Checkpoints with per-channel scales ("FP8-dynamic") or
  128-element block scales are converted to a 16-bit type, and the patch does not apply to them.
- If the machine has more than one ROCm device (for example an integrated GPU), set `HIP_VISIBLE_DEVICES` so llama.cpp
  sees only the discrete card.
- Always pass `-c` to `llama-completion`. Without it llama.cpp sizes the context to the free VRAM (89,344 tokens for the
  8B model, about 11 GB of KV cache), and it ran my machine out of memory once.

### dxil-spirv (experimental)

[patches/dxil-spirv-rdna3-exact-fp8.patch](patches/dxil-spirv-rdna3-exact-fp8.patch) applies to dxil-spirv `114e664`, the
commit pinned by vkd3d-proton `5b1089c8`. It touches 5 files, about 200 lines. Select an option with the
`DXIL_SPIRV_CONFIG` environment variable: `wmma_rdna3_exact_fp8` or `wmma_rdna3_exact_fp8_rte`. Not tested in a game, see
section g.

## Limits

- One GPU (RX 7900 XT). Other RDNA 3 cards are untested.
- Two models, both Llama (3B and 8B). Other models, sizes, mixture-of-experts shapes and Vulkan are untested. RDNA 4 is
  not a target: it has native FP8 and a different WMMA layout.
- No independent review of the patches, and nothing sent upstream.
- FSR 4: the dxil-spirv patch is untested in a game, so I have no evidence about image quality or speed.
- The llama.cpp NaN bug is not yet reported upstream. The same overflow mechanism is behind an open issue for other types
  on CUDA: ggml-org/llama.cpp#30951.
- The prompt kernel handles shapes where the number of rows is a multiple of 128 and the row length is a multiple of 64.
  Others use the slower fallbacks. The small K/V layers (1024 x 3072) are about 10 % behind hipBLAS in isolation.
- GPU memory was measured only for the 3B model.
- Timings come from a machine I also used for other things. Run-to-run spread was a few percent. Compare ratios, not
  single numbers.

## How this was made

The investigation was mine: I chose the question, set the goals and stop points, and decided what to keep. AI agents
(Anthropic Claude models) did the coding, ran the measurements and drafted the text, working under my direction. I
checked their results against the code and reran the tests. That does not make the work reviewed by anyone else. Treat
the numbers as measurements from one machine, and read the patches before relying on them.

## Licence and credits

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 j0nsh1n.

- The patches apply to llama.cpp (MIT) and dxil-spirv (MIT). Those projects keep their own licences.
- Models are not included. I used `RedHatAI/Llama-3.2-3B-Instruct-FP8` and `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`,
  under the Llama community licences.
- The idea of widening FP8 weights to FP16 inside a matrix kernel is not new. FP8 Marlin (vLLM) does it on NVIDIA Ampere,
  and feather-ops already does it with WMMA on gfx1151 (RDNA 3.5) for image models, the same design as my prompt
  kernel. What I add is the decode kernels and llama.cpp integration for LLMs, the overflow fix, the exactness checks
  and the measurements.
