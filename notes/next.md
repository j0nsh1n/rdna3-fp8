# Next steps

1. Survey (read-only): llama.cpp (ROCm / HIP and Vulkan backends), vLLM on ROCm, PyTorch on ROCm (float8 tensors,
   scaled matmul), ONNX Runtime / MIGraphX, MLC-LLM and others that run on RDNA 3. For each: does it load FP8
   checkpoints on gfx11, and how (convert at load, refuse, emulate in kernels); FP8 KV cache support and its method;
   who maintains the ROCm path and whether outside kernels are accepted.
2. Decide with the user whether a prototype is worth it: an FP8-weight GEMV / GEMM kernel on the 7900 XT, FP8 bytes
   in VRAM widened in registers, timed against FP16 weights at decode-like and prefill-like shapes.

Rules carried over from the WrongGPU work: every helper-built program runs under a memory cap
(`systemd-run --user --scope -q -p MemoryMax=... -- nice ...`); GPU runs are headless and short; timed runs take the
lock `flock <project>/gpu-bench.lock` (shared with other GPU jobs on this machine through the WrongGPU build folder's
lock if both run: build/rdna3-analysis/gpu-bench.lock in the wronggpu project).
