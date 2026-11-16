# Measurements

Full tables behind the "Architecture limitations" section of the [top-level README](../README.md). All were measured
on a Radeon RX 7900 XT (gfx1100).

| File | What it measures |
|---|---|
| [matrix-arithmetic.md](matrix-arithmetic.md) | How the WMMA instructions add and round, against 13 arithmetic models. |
| [errors-after-quantizing.md](errors-after-quantizing.md) | How often that rounding changes an FP8 or FP16 result. |
| [shader-throughput.md](shader-throughput.md) | Cost per value of widening, rounding and lookup operations in shaders. |
| [e4m3-rounding.md](e4m3-rounding.md) | Exact FP32 to E4M3 rounding without FP8 hardware, with the variants that failed. |

These files come from the WrongGPU project (MIT, https://github.com/phantoslayer21/wronggpu) and are copied
unchanged. File paths inside them (`run.sh`, `out/`, `logs/`, `shaders/` and similar) point to that project's lab
folders, which are not part of this repository.
