# Model test 1: Llama 3.2 3B Instruct FP8 on the RX 7900 XT

Date: 2026-11-15. Model: RedHatAI/Llama-3.2-3B-Instruct-FP8 (Llama 3.2 Community License), compressed-tensors
`float-quantized`, one scale per weight tensor, `lm_head` not quantized. Downloaded to `models/` (gitignored).
llama.cpp 6d3568e31 with `bench/llama-cpp-fp8-mmvf-hip.patch` (patched) or without (`build/llama.cpp-base`).
Tools: `bench/run-model.sh`, `proto/gguf_output_q8.py`, `proto/parse_model_runs.py`. Every GPU run on device 0 only
(llama.cpp otherwise also sees the integrated GPU) and under the shared GPU lock.

## In short

- Unmodified llama.cpp gives NaN for FP8 models on RDNA 3 whenever more than 8 tokens are processed at once
  (prompts, perplexity). Fixed in the patch.
- With the patch, answers match unmodified llama.cpp's exact path to rounding level, and FP8 is faster than Q8_0 for
  one to seven parallel sequences; 10 % slower at eight; 13 % slower at prompt processing.

## Files

| GGUF | Transformer weights | Output layer | Size |
|---|---|---|---|
| llama32-3b-auto | F8_E4M3 (all 7 matrices, 28 layers) | BF16 (as shipped) | 4.4 GB |
| llama32-3b-fp8-outq8 | F8_E4M3 | Q8_0 (`gguf_output_q8.py`; all else byte-identical) | 4.0 GB |
| llama32-3b-q8_0 | Q8_0 (from the F16 GGUF) | Q8_0 | 3.8 GB |
| llama32-3b-f16 | F16 (converted from the FP8 checkpoint) | F16 | 7.2 GB |

The matched variant keeps FP8 from being charged for a 16-bit output layer (788 MB read per token as BF16, 419 MB as
Q8_0) that has nothing to do with the FP8 kernels.

## The NaN bug in unmodified llama.cpp

For more than 8 columns, `ggml_cuda_mul_mat_cublas` converts F8 weights to F16 and, on GPUs with fast F16, computes
with `CUBLAS_COMPUTE_16F` and F16 output; only Volta, RDNA 4 and CDNA get F32 output (`prefer_f32_output`). FP8
weights are unscaled there (up to 448; the per-tensor scale, about 1e-3 here, is applied by a later MUL), so outputs
overflow F16 and become inf, then NaN. Evidence (unpatched, 4 chunks): CPU 9.7036; GPU with 8-token batches (mmvf,
F32) 9.7038; GPU 512-token batches NaN; same with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` 9.7032. Weights have no NaN
codes and finite scales. llama.cpp's own tests miss it because their F8 test weights lie in [-1, 1].

Fix in the patch (`ggml-cuda.cu`):
1. Fast path: the existing MUL_MAT + scale MUL fusion hook (`ggml_cuda_mul_mat_fp8`, a stub on HIP) now has a HIP
   sibling, `ggml_cuda_mul_mat_fp8_scaled_f16`: convert the F8 weight to F16 with the scale folded in (shift
   widening, NaN codes kept as NaN, 8-byte stores), then the regular F16 cuBLAS path writes the MUL's output.
2. Fallback for anything the hook does not take: F8 weights get F32 output in `ggml_cuda_mul_mat_cublas_impl`
   (correct but slow on RDNA 3: prompt processing 1850 t/s).

## Correctness

- `test-backend-ops test -b ROCm0 -p f8_e4m3`: 450/450 OK on the final build.
- Perplexity, 50 chunks of a fixed local text (llama.cpp docs, 475 KB): FP8 (patched) 9.477 (scale-folding path),
  Q8_0 9.474, F16 9.494; unpatched FP8 NaN. With the F32-output fallback instead: 9.462.
- KL divergence, 10 chunks, against unpatched FP8 with 8-token batches (llama.cpp's own F32 decode path):

| Patched run | Mean KLD | Same top token |
|---|---|---|
| 1-, 3- and 8-token batches (our decode kernels) | 0.000002-0.000004 | 99.92-99.96 % |
| 512-token batches (scale-folding F16 path) | 0.00108 | 97.6 % |
| F16 model, unmodified F16 path, for comparison | 0.00206 | 96.7 % |
| Q8_0 model, for comparison | 0.00061 | 98.4 % |

The prompt path inherits llama.cpp's F16 GEMM precision on RDNA 3 (F16 accumulation); it is closer to the reference
than the F16 model on the same path, further than Q8_0.

## Speed (final build, run build/model-runs/20261115-145358 and earlier)

Single stream, `llama-bench` tg128 (t/s, median of 3 rounds): Q8_0 145.0, FP8 matched 157.8, FP8 as shipped 146.8,
F16 94.5; unpatched FP8 66.2. Prompt pp512: Q8_0 7560-7716, F16 7314-7463, FP8 6634-6640 (scale-folding),
unpatched FP8 about 6300 but NaN output.

Parallel sequences, `llama-batched-bench`, decode t/s (all sequences), 2 rounds:

| Sequences | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 142.8 | 261.5 | 379.0 | 488.9 | 586.7 | 675.8 | 763.5 | 845.8 |
| FP8 patched | 156.6 | 280.8 | 405.0 | 524.7 | 621.3 | 706.0 | 771.2 | 758.6 |
| FP8 / Q8_0 | 1.10 | 1.07 | 1.07 | 1.07 | 1.06 | 1.04 | 1.01 | 0.90 |
| FP8 unpatched | 65.7 | 113.8 | 161.1 | 201.9 | 241.4 | 273.7 | 271.5 | 297.1 |

Tuning that got there: this model's matrices are small (K and V are 1024 x 3072), so the step 3 kernel left most of
the card idle. The rows kernel now lets 2 warps split each row's length (`ksplit`, partial sums through LDS), with 4
rows per warp for 2-7 columns and 2 for 8, chosen by a rows x ksplit sweep in the model
(`build/runs/sweep-batched-*.log`) and checked on the 16384 x 14336 weight, where FP8 is now 0.91-0.97x Q8_0's time
at every n = 1..8 (run build/runs/20261115-145158).

## Open

- Eight parallel sequences: 10 % behind Q8_0 in this model (ahead on the big weight).
- Prompt processing: 13 % behind Q8_0 because every 512-token batch converts the weights again; a dedicated FP8
  prompt kernel (prototype at 72 % of Q8_0 on the big weight) would remove the conversion.
- Not yet done: an actual generation test with a real prompt (owner asked for it after the tuning).
