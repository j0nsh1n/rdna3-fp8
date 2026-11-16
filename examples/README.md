# Examples

Small standalone programs behind the numbers in the [top-level README](../README.md). They are prototypes and
measurement tools, not a library. The production kernels are in
[../patches/llama.cpp-fp8-rdna3.patch](../patches/llama.cpp-fp8-rdna3.patch).

Common rules:

- Run all commands below from this `examples/` folder.
- Build with ROCm's `hipcc`: `hipcc -O3 -std=c++17 --offload-arch=gfx1100 <file> -o <name> -lamdhip64`. Some
  distributions' `hipcc` (Fedora's, for one) do not link the HIP runtime on their own, hence `-lamdhip64`.
- The GPU programs read device 0 and refuse to run unless it is a gfx1100 card (`device 0 is ..., not the gfx1100
  card`). On a machine with an integrated GPU, set `HIP_VISIBLE_DEVICES=0` (or the index of the discrete card) first.
- For timings, run when the GPU is otherwise idle. The numbers in the README have a spread of a few percent.

## widen-check/widen_check.cpp

CPU program. Checks the E4M3 to FP16 bit-shift widening and the per-word NaN-code check against llama.cpp's own
conversion, for all 2^32 input words and every byte position. It prints the number of mismatches and exits with 0 only if there are none.

It needs llama.cpp's `ggml.h` and `libggml-base`. With a llama.cpp checkout in `$LLAMA` (absolute path) built into
`$LLAMA/build`:

```
g++ -O2 -std=c++17 -I$LLAMA/ggml/include widen-check/widen_check.cpp -L$LLAMA/build/bin -lggml-base \
    -Wl,-rpath,$LLAMA/build/bin -o widen_check
./widen_check
```

## decode-gemv/fp8_gemv.hip

The decode prototype (1 to 8 columns). Weights are E4M3 bytes, activations FP32. Variants:

- `a`: exact, FP32 activations, `v_fma_mix_f32`. Same products as llama.cpp.
- `b`: activations rounded to FP16, `v_dot2_f32_f16`. If any activation would overflow FP16, the kernel falls back to
  variant a's loop.
- `c`: variant a with each weight converted to FP32 first (the dual-issue experiment, which did not pay).

```
hipcc -O3 -std=c++17 --offload-arch=gfx1100 decode-gemv/fp8_gemv.hip -o fp8_gemv -lamdhip64
./fp8_gemv <a|b|c> <R 1|2|4> <N 1..8> [M K reps mode]
```

`R` is rows per warp, `N` the number of columns. `M K reps` default to 16384, 14336 and 200. `mode` is one of:

- `plain`: random weights, checks against a float64 reference on the CPU.
- `time`: timing only, skips the reference.
- `nan`: plants 0x7F in row 5 and 0xFF in the last row. Exactly those rows must come out NaN in every column.
- `ovf`: plants an activation of 1e5 (above FP16's 65504) in column 0. Variant b must fall back and stay exact.

To see the instruction counts, add `-save-temps` and run `tools/loop_isa.py` on the `.s` file.

## prompt-gemm/fp8_gemm.hip

The prompt (matrix-matrix) prototype: a WMMA kernel with many tile variants, plus a hipBLAS FP16 baseline in the same
harness for timing comparisons. It needs the hipBLAS headers and library.

```
hipcc -O3 -std=c++17 --offload-arch=gfx1100 prompt-gemm/fp8_gemm.hip -o fp8_gemm -lhipblas -lamdhip64
./fp8_gemm <N> [M K reps plain|nan|bench variant...]
```

`N` is the number of columns. `M` must be a multiple of 256 and `K` a multiple of 128 in this program (the kernel in the
patch needs only 128 and 64). Without `bench`, it checks the first variant you name (`v1` if none) against a CPU reference and times it. With
`bench` it first times hipBLAS on the same shape (FP16 and FP32 output), then the variants you name (`all` for every
variant). `nan` plants NaN codes as in the decode program.

## bandwidth/bandwidth.hip

Read-only and copy bandwidth of the card with 16-byte loads, as a ceiling for decode kernels. On the RX 7900 XT it gives
about 754 GB/s read and 603 GB/s copy with the default 1 GiB buffers.

```
hipcc -O3 -std=c++17 --offload-arch=gfx1100 bandwidth/bandwidth.hip -o bandwidth -lamdhip64
./bandwidth [MiB per buffer]
```

Use buffers much larger than the 80 MB Infinity Cache, or the numbers will be too high.

## pytorch/fp8_linear.py and pytorch/fp8_ext.hip

`FP8Linear`: a linear layer that keeps `float8_e4m3fn` weights in VRAM and calls the decode kernel from PyTorch. The
extension builds as a shared library and includes `../decode-gemv/fp8_gemv.hip` with `FP8_GEMV_NO_MAIN` defined, so it
has no `main`.

```
hipcc -O3 -std=c++17 --offload-arch=gfx1100 -fPIC -shared pytorch/fp8_ext.hip -o libfp8ext.so -lamdhip64
FP8EXT_LIB=$PWD/libfp8ext.so python3 -I pytorch/fp8_linear.py
```

`fp8_linear.py` expects the path of the library in the environment variable `FP8EXT_LIB`. It runs checks (against
float64, and the NaN rows) and timings at batch 1 to 8. It needs a PyTorch ROCm build. It imports torch before loading
the library so both use PyTorch's copy of `libamdhip64.so`, and it exits if it sees two HIP runtimes. Batch 1 to 8 only.

## tools/loop_isa.py

Counts instructions in every math loop of each kernel in an AMDGPU assembly file (`-save-temps` output). A math loop is
a backward-branch loop that contains `v_fma_mix_f32`, `v_dot2`, `v_wmma` or FP32 FMAs. For each loop it prints counts of
vector, scalar, memory and LDS instructions, vector instructions per weight, and the opcode histogram. I used it to catch
every compiler problem listed in the README before spending any GPU time.

```
python3 -I tools/loop_isa.py <file.s> [kernel-name-regex]
```

## quant/e4m3_rdna3.h

Exact FP32 to E4M3 rounding without FP8 hardware, from the WrongGPU project. Round to nearest even, saturate at 448,
bit-exact over all 2^32 inputs. It is written against that project's `wg_platform.h` types (`WG_FN`, `u32`, `v2i`, `v8f`,
`v8h`), so it does not compile on its own. Read it for the method (the magic-constant add and subtract) and port it, or
see [../measurements/e4m3-rounding.md](../measurements/e4m3-rounding.md) for the results and the variants that failed.
