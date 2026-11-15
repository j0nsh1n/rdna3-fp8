# PyTorch extension 1: FP8 weights in VRAM through our decode kernel

Date: 2026-11-15. PyTorch 2.13.0+rocm7.1 (`build/torch-venv`, gitignored). Code: `proto/torch/fp8_linear.py`
(the `FP8Linear` layer and the checks/timings), `proto/torch/fp8_ext.hip` (C entry point; builds the decode
prototype `proto/gemv/fp8_gemv.hip` as `build/torch-ext/libfp8ext.so`). Log: `build/torch-runs/fp8_linear-1.log`.
Baseline for comparison: `notes/pytorch-baseline-1.md` (native FP8 matmul raises on gfx1100).

## How it is wired

- `FP8Linear(weight_fp8, scale, variant)` keeps the `torch.float8_e4m3fn` weight in VRAM and calls
  `fp8_gemv_run` through ctypes with the tensors' device pointers and `torch.cuda.current_stream()`.
- PyTorch ships its own `libamdhip64.so` with the same soname (`libamdhip64.so.7`) as the system one; importing torch
  before loading our library makes both use PyTorch's copy. The script checks `/proc/self/maps` and refuses to run
  with two HIP runtimes mapped. Measured: one, PyTorch's.
- Input is converted to F32 and the F32 output back to the input dtype; both conversions are inside the timings.
- Rows per wave per batch size are the prototype's best (`BEST_R`). Batch 1..8 only.

## Checks (all passed)

- First 64 output rows against float64, FP16 input and output, n = 1, 4, 8: max |err| / max |ref| 3.1e-4 to 3.3e-4
  for both variants (dominated by the FP16 output rounding, bound 2^-11 = 4.9e-4).
- A 0x7F byte in weight row 5: row 5 is NaN in every column, no other row has NaN, both variants.

## Timing (16384 x 14336, FP16 input/output, 200 calls after 20 warm-up, median us, min-max)

| n | PyTorch F.linear, FP16 weights | FP8 layer A (exact) | FP8 layer B (FP16 activations) |
|---|---|---|---|
| 1 | 836.2 (778-1024) | 321.8 (315-382) | 329.6 (320-360) |
| 2 | 819.9 (790-973) | 326.0 (318-402) | 332.0 (323-349) |
| 3 | 846.4 (787-998) | 343.7 (334-397) | 331.7 (324-378) |
| 4 | 856.4 (795-1015) | 356.0 (347-487) | 342.4 (335-413) |
| 5 | 869.5 (798-1023) | 368.2 (353-492) | 341.4 (334-409) |
| 6 | 853.2 (789-1140) | 428.3 (372-522) | 374.9 (358-475) |
| 7 | 945.1 (819-1192) | 508.2 (443-615) | 394.7 (353-495) |
| 8 | 946.6 (837-1109) | 500.1 (434-573) | 403.1 (357-547) |

- One locked run on a shared card; F16 times vary more here than in the baseline (826-945 us there).
- The FP8 layer is 2.3-2.6x faster than PyTorch's FP16 path with half the weight memory, and 8-14x faster than
  PyTorch's per-call FP8 conversion (baseline notes).
- Variant A reaches about 500 us at n = 7, 8 here too, matching llama.cpp's harness (notes/llama-cpp-integration-1.md):
  three harnesses now agree that A is instruction-bound there.
- Not done: prefill (n > 8), a `torch.nn.Module` wrapper with autograd, `torch.compile` integration.
