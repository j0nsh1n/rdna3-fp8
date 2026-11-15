# Handoff: rdna3-fp8 (2026-11-15)

Written by the Claude session that did the RDNA 3 analysis for WrongGPU. The owner wants a separate Claude session to
run this project from here.

## Goal

Make FP8 AI models cheaper to run on AMD RDNA 3 GPUs (Radeon RX 7000, gfx11), which have no FP8 instructions:
keep FP8 weights as 8-bit bytes in VRAM and widen them to FP16 in registers (instead of converting the whole model to
16-bit at load), and use an exact, fast FP32 -> FP8 (E4M3) rounding where runtimes write FP8 (KV caches).

## What exists

- `README.md`: the goal.
- `carry-over/`: measured on this machine's Radeon RX 7900 XT (gfx1100, 20 GB) during the WrongGPU work:
  `quant/e4m3_rdna3.h` (exact E4M3 rounding in HIP: 7.7 cycles/value as FP16 output, 11.3 as code bytes, bit-exact over
  all 2^32 inputs), `quant/RESULTS.md`, and `measurements/` (F16 and IU8 matrix instructions both 32 cycles per
  16x16x16; packed int8 dot at half that rate; matrix and ordinary instructions never overlap; the F16 matrix
  instruction's sums are slightly inexact for mixed signs).
- `notes/survey.md`: read-only survey of llama.cpp, vLLM, PyTorch, SGLang, ONNX Runtime, MLC-LLM and others.
  Headline: llama.cpp has a new `GGML_TYPE_F8_E4M3`, but on HIP decode converts FP8 to FP32 in generic software and
  prefill converts weights to FP16; vLLM most likely refuses FP8 checkpoints on gfx1100; feather-ops (MIT) already
  widens FP8 to FP16 for WMMA on RDNA 3 but targets diffusion on Strix Halo. The catch: llama.cpp's Q8_0 already
  gives 8-bit weights on RDNA 3 with mature kernels, so FP8 storage only matters if a kernel beats Q8_0. Gaps in the
  survey: Composable Kernel and hipBLASLt gfx11 kernels (rate-limited), nothing was run.

## Next step (agreed with the owner)

A go / no-go measurement before writing any kernel:
1. Build llama.cpp locally with HIP for gfx1100 (ROCm 7.1.1 from the distribution is installed: `hipcc`, `rocminfo`;
   linking HIP programs needs `-L/usr/lib64 -lamdhip64`).
2. One small open model (1-3 B parameters) in three forms: Q8_0, F16, and llama.cpp's FP8 type if its converter can make
   it. Time token generation (batch 1) and prompt processing with `llama-bench`.
3. Measure the card's copy bandwidth and express each decode result as a share of it.
4. Decide with the owner: if Q8_0 is near the bandwidth limit and the FP8 path is close to it, stop or pick another
   target (vLLM's missing gfx11 FP8 kernel). If the FP8 path is well below Q8_0, prototype a standalone HIP decode GEMV
   (16-byte loads, widen in registers, `v_dot2_f32_f16`) and a WMMA prefill GEMM, and time them against the same
   baselines.

## Rules that came with this work (keep them)

- This machine is the owner's desktop: he works and games on it. Run every program you build (and builds, benchmarks)
  under a memory cap: `systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- nice <command>`. A helper
  once ran a benchmark to 16 GB and triggered the out-of-memory killer. Check `free -g` before heavy jobs.
- GPU runs are headless and short. Never open windows. Scratch goes in the project folder or `~/Projects/...`, never
  `/tmp` (memory-backed here).
- Another project (WrongGPU) also times things on this GPU. Wrap every timed GPU run in
  `flock /home/jonathans/Projects/wronggpu/build/rdna3-analysis/gpu-bench.lock <command>` so the two never overlap.
- Git: this repo's local identity is the owner's GitHub no-reply address; keep it. Never commit his personal email.
  No remote exists. Never push, create a remote or open PRs or issues anywhere without his word.
- Downloads: model files only from their official pages, into a folder under this project that is gitignored; say
  which model and licence before downloading.
- After the owner interrupts a turn, check any helper you started in that turn before saying it is running (an
  interrupt can end it silently).
