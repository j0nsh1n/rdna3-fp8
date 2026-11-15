# Go / no-go measurement 1: llama.cpp FP8 vs Q8_0 and F16 on the RX 7900 XT

Date: 2026-11-15. Radeon RX 7900 XT (gfx1100, 20 GB, 80 MB Infinity Cache), Linux, ROCm/HIP 7.1.1 from the
distribution. Every run was headless, memory-capped and under the shared GPU lock. Each number below is one
`test-backend-ops perf` run (it repeats each case hundreds to thousands of times and reports the mean).

## Setup

- llama.cpp `6d3568e31` (the commit the survey read; tag b12408), built with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
  -DCMAKE_BUILD_TYPE=Release` in `build/llama.cpp/build-hip` (gitignored). The build needed the distribution packages
  `hipblas-devel` and `rocblas-devel` (installed 2026-11-15).
- Correctness first: `test-backend-ops test -b ROCm0 -o MUL_MAT -p type_a=f8_e4m3` passes all 91 supported cases
  against the CPU. FP8 weights with F16 activations are "not supported" on HIP; the model path uses F32 activations.
- Card bandwidth: `bench/bandwidth.hip`, 1 GiB buffers, 16-byte loads, 50 repetitions: **read 754 GB/s** (median;
  best 766), copy 603 GB/s (counting read + write). Read is the ceiling for a decode GEMV. Vendor spec is 800 GB/s.
- Shape: weight 16384 x 14336, activations F32, batch n = 1, 4, 8 (decode) and 512 (prompt). llama.cpp's own perf
  shape (4096 x 14336) is too small: its 8-bit weights (about 60 MB) stay in the 80 MB Infinity Cache between
  repetitions, which showed Q8_0 "reading" at 1737 GB/s. The larger case is added locally by
  `bench/llama-cpp-out-of-cache-shape.patch`; its weights are 132-470 MB.

## Results (weight bytes read per run / time; share of the 754 GB/s read ceiling)

| Weights | n=1 | n=4 | n=8 | n=512 time |
|---|---|---|---|---|
| F16 | 636 us, 739 GB/s (98 %) | 757 us, 82 % | 770 us, 81 % | 4245 us |
| BF16 | 640 us, 97 % | 755 us, 82 % | 784 us, 79 % | 4310 us |
| Q8_0 | 346 us, 721 GB/s (96 %) | 386 us, 86 % | 483 us, 68 % | 4202 us |
| **F8_E4M3** | **1099 us, 214 GB/s (28 %)** | **1210 us, 26 %** | **1669 us, 19 %** | **5147 us** |
| Q4_0 | 190 us, 92 % | 225 us, 78 % | 379 us, 46 % | 4092 us |

The 4096 x 14336 run (in cache) agrees in direction: F8 198 us, F16 169 us, Q8_0 36 us at n=1.

## What it means

- llama.cpp's FP8 decode on gfx1100 is limited by its software conversion (one byte at a time to F32, see the
  survey section 1.2), not by memory: 28 % of the ceiling, and **1.7x slower than F16, which reads twice the bytes**.
  Against Q8_0, same byte count, it is 3.2x slower at n=1 and 3.5x at n=8.
- The method can show near-full bandwidth (F16 98 %, Q8_0 96 %), so the low FP8 number is the kernel, not the probe.
- Prompt processing (n=512) is compute-bound for all types; FP8 is 21 % slower than F16 there (the extra
  dequantize-to-temporary pass before hipBLAS).
- The ceiling for a fixed FP8 decode is Q8_0's speed, about 6 % faster at most (1 byte per weight against 1.0625).
  So a kernel would make FP8 GGUFs as fast as Q8_0 on RDNA 3; it would not make RDNA 3 faster than Q8_0 already is.

Against the survey's stop criteria (section 8): "F8 decode already >= ~85 % of bandwidth" is false (28 %).
"Q8_0 matches FP8 on speed" is true for what is possible: Q8_0 is already at 96 %. The open question is the owner's:
is "FP8 checkpoints run at Q8_0 speed in llama.cpp on AMD" worth a kernel and an upstream PR.

## Not measured

- A real model with `llama-bench` (needs an FP8 checkpoint with one scale per tensor; the quantize tool cannot make
  F8_E4M3 from a 16-bit model).
- Whether RDNA 4 and CDNA 3, which have FP8 hardware, also take the software path in llama.cpp's HIP build. The source
  says yes (the hardware branch is `!defined(GGML_USE_HIP)`); no such card here.
- Repeat runs for spread; the gaps above are 3x, far beyond run-to-run noise seen so far, but that was not measured.
