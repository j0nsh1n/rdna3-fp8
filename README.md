# rdna3-fp8

Running FP8 work on AMD RDNA 3 GPUs (Radeon RX 7000, gfx11), which have no FP8 instructions. Target questions:

1. Can local-AI runtimes keep FP8 model weights in 8 bits in VRAM on RDNA 3 and widen them to FP16 in registers for
   the FP16 matrix or dot-product instructions, instead of converting the whole model to FP16 or BF16 at load (twice
   the memory)? Does it make token generation faster, since that is limited by memory bandwidth?
2. Where can an exact, fast FP32 -> FP8 (E4M3) rounding help, for example an FP8 attention (KV) cache?

Status: research. First step is a read-only survey of which runtimes run on RDNA 3 and how they handle FP8 today.

## Starting material (`carry-over/`)

Measured on a Radeon RX 7900 XT (gfx1100), Linux, during the RDNA 3 work for WrongGPU
(https://github.com/phantoslayer21/wronggpu, issue #28):

- `quant/e4m3_rdna3.h`: exact FP32 -> E4M3 rounding in HIP, 7.7 cycles per value as FP16 output and 11.3 as code bytes
  (bit-exact over all 2^32 inputs against a saturating round-to-nearest-even reference); `quant/RESULTS.md` has the
  method and timings.
- `measurements/`: the F16 and IU8 matrix instructions' arithmetic and timing, shader-instruction throughput, and how
  often the F16 instruction's small errors change a value after rounding to FP8.

These files were written for that analysis and are MIT licensed like WrongGPU.
