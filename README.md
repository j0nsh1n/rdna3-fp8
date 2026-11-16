# rdna3-fp8

FP8 (E4M3) AI models on AMD RDNA 3 GPUs (Radeon RX 7000), which have no FP8 instructions.

RDNA 3 cannot do FP8 math, but it can store FP8 weights in 8 bits and widen them to FP16 in registers. This repo does
that inside llama.cpp, as a patch, and tests it on a Radeon RX 7900 XT (gfx1100). Unmodified llama.cpp is slow with FP8
models on this GPU, and it prints garbage ("GGGG...") for any prompt longer than 8 tokens because of a bug. With the
patch, the same FP8 files give correct answers and run faster than the 8-bit Q8_0 format for one to eight parallel
streams and for prompt processing. The repo also holds prototype kernels, a PyTorch extension, and an experimental patch for
the FSR 4 emulation layer (dxil-spirv) that is not yet tested in a game.

Status: research. Everything was measured on one GPU. Nothing has been sent to llama.cpp or dxil-spirv upstream.

## Results

All numbers are from an RX 7900 XT with ROCm/HIP 7.1.1, llama.cpp commit `6d3568e31` with and without
`bench/llama-cpp-fp8-mmvf-hip.patch`. "Q8_0" is the same weights re-rounded to llama.cpp's 8-bit format. The FP8 file
used here has its output layer converted to Q8_0 (`proto/gguf_output_q8.py`), so the comparison measures the FP8
weights and not a 16-bit output layer. Tokens per second (t/s); higher is better. Details are in
[notes/model-test-1.md](notes/model-test-1.md) and [notes/model-test-8b.md](notes/model-test-8b.md).

### Llama 3.2 3B Instruct FP8

| What | Q8_0 | FP8, patched | FP8 vs Q8_0 | FP8, unmodified llama.cpp |
|---|---|---|---|---|
| One stream, generation | 145.0 t/s | 157.8 t/s | +9 % | 66.2 t/s |
| 2-8 parallel streams | | | +3 % to +7 % | 2.2 to 2.9x slower than Q8_0 |
| Prompt processing (512 tokens) | 7660-7678 t/s | 8545-8962 t/s | +12 % to +17 % | wrong output (NaN) |
| Prompt processing (2048 tokens, `-ub 2048`) | 7563-7657 t/s | 9213-9225 t/s | +20 % | wrong output (NaN) |
| GPU memory, peak in a run | 4369 MiB | 4175 MiB | -4 % | |
| Answer quality (perplexity, lower is better) | 9.474 | 9.463 | same or slightly better | NaN |

Notes: the one-stream and 512-token prompt numbers come from the final build. The parallel-stream ratios are from the
final decode kernel ([notes/n8-gap-1.md](notes/n8-gap-1.md), 3 rounds of `llama-batched-bench`). In two real prompts
(a refrigerator explanation and a Python prime check) the FP8 and Q8_0 replies were word for word identical. Unmodified
llama.cpp printed only "GGGG...".

### Llama 3.1 8B Instruct FP8

| What | Q8_0 | FP8, patched | FP8 vs Q8_0 | FP8, unmodified llama.cpp |
|---|---|---|---|---|
| One stream, generation | 76.3 t/s | 81.9 t/s | +7 % | 30.3 t/s |
| 2-8 parallel streams | | | +6 % to +7 % | 2.7 to 3.3x slower than Q8_0 |
| Prompt processing (512 tokens) | 3469.6 t/s | 4048.5 t/s | +17 % | wrong output (NaN) |
| GPU memory | not measured | not measured | | |
| Answer quality (perplexity, lower is better) | 7.2763 | 7.2655 | same or slightly better | NaN |

Notes: with greedy decoding the FP8 and Q8_0 replies differ in wording on this model (both are proper answers).
Q8_0 here is a re-rounding of the FP8 weights, so a tie between two words can go either way.

### Before this work

On unmodified llama.cpp, FP8 models on RDNA 3 are 2.2 to 3.5x slower than Q8_0 at generation, and 2.4 to 2.7x slower
than the patched FP8 path at one stream. For prompts longer than 8 tokens they produce NaN: perplexity is NaN and generated text is
"GGGG..." only. Section "The llama.cpp NaN bug" below has the cause.

## What is in the repo

| Folder | Contents |
|---|---|
| `bench/` | Patches and scripts. |
| `proto/` | Prototype kernels, a PyTorch extension, and checking tools. |
| `carry-over/` | Exact FP32 to E4M3 rounding and RDNA 3 measurements from the WrongGPU project. |
| `notes/` | Lab notes with the full measurements. |

Main files in `bench/`:

- `llama-cpp-fp8-mmvf-hip.patch`: the llama.cpp patch (decode kernels, prompt kernel, NaN fix).
- `llama-cpp-convert-naive-fp8.patch`: one-line fix so llama.cpp's converter accepts `naive-quantized` checkpoints.
- `llama-cpp-out-of-cache-shape.patch`: adds a larger benchmark case to `test-backend-ops` (benchmark only).
- `dxil-spirv-rdna3-exact-fp8.patch`: experimental option for dxil-spirv (see the FSR 4 section).
- `run-gpu.sh`: kernel correctness and timing. `run-model.sh`: perplexity, KL divergence and speed on a model.
  `fsr4-game.sh`: file switch script for the FSR 4 test. `bandwidth.hip`: memory bandwidth probe.

Main files in `proto/`:

- `gemv/fp8_gemv.hip`: decode kernel prototypes. `gemv/loop_isa.py`: counts instructions in a compiled loop.
- `gemm/fp8_gemm.hip`: prompt (matrix-matrix) kernel prototypes with a hipBLAS baseline.
- `widen_check.cpp`: exhaustive check of the E4M3 to FP16 widening over all 2^32 input words.
- `torch/`: `FP8Linear` PyTorch layer that keeps FP8 weights in VRAM (`fp8_linear.py`, `fp8_ext.hip`).
- `gguf_output_q8.py`: makes a GGUF copy with a Q8_0 output layer. `parse_*.py`: log parsers.

Notes: [FINDINGS.md](FINDINGS.md) is the technical summary. The numbered files in `notes/` are the underlying
measurements: [go-no-go-1](notes/go-no-go-1.md), [prototype-results-1](notes/prototype-results-1.md),
[llama-cpp-integration-1](notes/llama-cpp-integration-1.md), [model-test-1](notes/model-test-1.md),
[model-test-8b](notes/model-test-8b.md), [prompt-test-1](notes/prompt-test-1.md),
[pytorch-baseline-1](notes/pytorch-baseline-1.md), [pytorch-ext-1](notes/pytorch-ext-1.md), and the starting
[survey](notes/survey.md).

## How to use the llama.cpp patch

The patch is written against llama.cpp commit `6d3568e31` (tag b12408). On other commits it may not apply.

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 6d3568e31
git apply /path/to/rdna3-fp8/bench/llama-cpp-fp8-mmvf-hip.patch
# only for compressed-tensors "naive-quantized" checkpoints (for example the 8B model below):
git apply /path/to/rdna3-fp8/bench/llama-cpp-convert-naive-fp8.patch
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build needs the ROCm development packages, including hipBLAS and rocBLAS headers (`hipblas-devel`,
`rocblas-devel` on Fedora). The prompt kernel only runs on RDNA 3 (`gfx1100` and relatives built with the RDNA3
define). Other GPUs keep the old behaviour.

Which FP8 checkpoints stay FP8: llama.cpp's converter keeps FP8 weights only when each tensor has one scale (a
single value). Checkpoints with per-channel scales ("FP8-dynamic") or 128-element block scales are converted to a
16-bit type at conversion time and the patch does not apply to them.

If your machine has more than one ROCm device (for example an integrated GPU), set `HIP_VISIBLE_DEVICES` so llama.cpp
sees only the discrete card. Always pass a context size (`-c`) to `llama-completion`: without it llama.cpp sizes the
context to the free VRAM.

The scripts in `bench/` read machine settings from an optional `build/local.env` file (ignored by git):
`GPU_LOCK` (a lock file shared with other jobs that time the same GPU) and `CYBERPUNK_DIR` (the game folder for
`fsr4-game.sh`). Without it they use a lock file inside `build/`.

Models are not included. Models used:

- `RedHatAI/Llama-3.2-3B-Instruct-FP8` (compressed-tensors `float-quantized`, one scale per tensor)
- `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` (compressed-tensors `naive-quantized`, needs the converter patch)

## The llama.cpp NaN bug on RDNA 3

For more than 8 tokens at once, llama.cpp converts FP8 weights to FP16 and calls hipBLAS with FP16 output. Only
Volta, RDNA 4 and CDNA GPUs get FP32 output there. FP8 checkpoints store the weights unscaled (values up to 448) and
apply the per-tensor scale (about 1e-3 in the 3B model) in a later step, so the FP16 output goes past 65504, becomes
infinity, then NaN. Their test weights lie in [-1, 1], which is why the bug is not caught. Evidence: on the same file
and text, perplexity is 9.7036 on CPU, 9.7038 on GPU with 8-token batches, NaN with 512-token batches, and 9.7032
when the output type is forced to FP32.

Status: not yet reported upstream. The patch fixes it locally in three layers: a dedicated FP8 prompt kernel where
the shape fits; otherwise the per-tensor scale is folded into the FP16 conversion; anything left uses FP32 output. The
same FP16 overflow mechanism is known for other types on CUDA (open issue ggml-org/llama.cpp#30951).

## FSR 4 and dxil-spirv (experimental)

FSR 4 can be run on RDNA 3 through vkd3d-proton and dxil-spirv, which emulate its FP8 matrix instructions with FP16.
Reading dxil-spirv (commit 114e664) showed that the emulation does not round values to the FP8 grid where RDNA 4
would: it converts FP32 to FP16 and only clamps to +-448. So on RDNA 3 the values passed between FSR 4's layers keep
FP16 precision (11 bits) instead of FP8's 4 bits. That is a different (more precise) network from the one AMD ships
for RDNA 4, not an approximation of it. Details are in [FINDINGS.md](FINDINGS.md).

`bench/dxil-spirv-rdna3-exact-fp8.patch` adds two `DXIL_SPIRV_CONFIG` options, `wmma_rdna3_exact_fp8` and
`wmma_rdna3_exact_fp8_rte`, that round to the E4M3 grid exactly (using the method in `carry-over/`). With the option
unset the output is unchanged. All 30 AGS test shaders in dxil-spirv's own test suite validate in every mode.

**Not yet tested in a game.** No in-game run has happened, so there is no evidence yet about image quality or speed.
`bench/fsr4-game.sh` is a snapshot, switch and restore script for such a test.

## Limitations and open items

- One GPU model tested (RX 7900 XT, 20 GB). Other RDNA 3 cards are untested.
- Two models tested, both Llama. Other models, sizes, mixture-of-experts shapes and Vulkan are untested.
- The prompt kernel handles shapes where rows are a multiple of 128 and the row length a multiple of 64; others
  use slower fallbacks. The smallest layers (1024 x 3072) are about 10 % behind hipBLAS in isolation.
- RDNA 4 is not a target. It has native FP8 and a different matrix layout.
- GPU memory was measured only for the 3B model.
- The patches have had no independent review and have not been sent upstream.
- The dxil-spirv patch is untested in a game.

## A note on how this was made

The investigation, code and write-ups were done by AI agents (Anthropic Claude models), directed and reviewed by
the repository owner. The measurements were run on the owner's hardware. Treat the numbers as measurements from one
machine, and check the patches before relying on them.

## Licence

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 j0nsh1n.

- The patches apply to llama.cpp (MIT) and dxil-spirv (MIT). Those projects keep their own licences.
- `carry-over/` comes from the WrongGPU project (MIT, https://github.com/phantoslayer21/wronggpu).
- Model files are not included. The models named above are under the Llama community licences.
