# Survey: FP8 on RDNA 3 (gfx11) in local-AI runtimes

Date: 2026-11-15. Read-only; no GPU program was run. Labels: **READ** = seen in a source (link given),
**INFERRED** = my reasoning from READ facts, **UNKNOWN** = not found or not checked. Commit/tag names are the ones
read via `gh api`; "master/main@sha" = the tip on 2026-11-15.

Pinned states: llama.cpp master@`6d3568e31d` (tag v1.0.0, 2026-11-13); vLLM main@`9bc16ae6a3` (latest release
v0.33.0rc3, 2026-11-02); PyTorch main@`48f1cea6cf` (v2.15.0, 2026-10-28); SGLang main@`2e55a173a7` (v0.5.23);
ONNX Runtime main@`1f53982117` (v1.32.0, 2026-11-13); MLC-LLM main@`3b7663abb6`; ExLlamaV3 master@`8c593e4e5c`;
feather-ops master@`0e08b4e223`; ComfyUI-FeatherOps master@`37e71445e7`.

Limits of this survey: the GitHub code-search API rate-limited me near the end, so Composable Kernel (CK) and
hipBLASLt kernel contents for gfx11 were not read (UNKNOWN). AMD Quark was checked only by web search.

## 0. Hardware baseline (what "no FP8 on gfx11" means in docs)

- READ: AMD's precision table lists low-precision float types (FP4/FP6/FP8) as unsupported in matrix cores for RDNA3
  and supported for RDNA4 (OCP E4M3 column) and CDNA3/4:
  https://rocm.docs.amd.com/en/latest/reference/precision-support.html (fetched 2026-11-15; the HIP type table shows
  `__hip_fp8_e4m3` supported only on CDNA4 and RDNA4; the FNUZ types only on CDNA3).
- READ: hipBLASLt lists FP8 types as library-supported but says this "does not indicate hardware support":
  https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html
- READ: PyTorch's `torch._scaled_mm` arch allow-list for ROCm is gfx942, gfx1200/1201 (ROCm >= 6.3), gfx950,
  gfx1250, and gfx1170/1171 (ROCm >= 10.3). gfx1100 is not in it:
  pytorch/pytorch@48f1cea6cf `aten/src/ATen/native/cuda/ScaledBlasDeviceUtils.cpp`, `scaled_mm_arch_allowed`.
  INFERRED: `_scaled_mm` fails on gfx1100 (I did not find the exact error string).
- READ: vLLM's own RDNA4 forum thread ("Native FP8 WMMA support for RDNA4",
  https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900) and a vLLM forum
  answer that the 7900 XTX is not supported for FP8 inference (https://discuss.vllm.ai/t/trying-to-run-gemma-3-27b-it-fp8-dynamic-with-rocm/1179)
  are consistent with this. Forum posts, indicative only.

## 1. llama.cpp (ROCm/HIP and Vulkan)

Maintainers: READ `CODEOWNERS`@6d3568e31d: `ggml-cuda` = am17an, gaugarg-nv, IMbackK, JohannesGaessler, ORippler,
thelittlefireman, ynankani; `ggml-hip` and `vendors/hip.h` = IMbackK, thelittlefireman; `ggml-vulkan` = 0cc4m,
jeffbolznv. The HIP backend is the CUDA backend compiled with `GGML_USE_HIP` (`ggml/src/ggml-cuda/`), so a kernel
added there serves both.

### 1.1 GGUF has an FP8 type now (new since mid-2026)
- READ: `GGML_TYPE_F8_E4M3 = 43`, `GGML_FTYPE_MOSTLY_F8_E4M3 = 29`, `ggml_fp8_e4m3_t` in
  `ggml/include/ggml.h`@6d3568e31d. OCP E4M3, saturating, no NaN preservation.
- READ: PR #27512 "Quant: OCP FP8 E4M3 support" (ORippler, merged 2026-10-29): https://github.com/ggml-org/llama.cpp/pull/27512
  Stack: #28392 FP8 KV cache (merged 2026-11-10), #28413 CUDA OCP FP8 (merged 2026-11-11), #28898 quant scales
  (open), #29791 Vulkan (open, updated 2026-11-14), #31871 SYCL (open). A CPU-only precursor #28485 and an older
  #10055 (2024) are open/stale. PR bodies disclose AI-assisted authorship (policy in `CONTRIBUTING.md`/`AGENTS.md`).
- READ: `convert_hf_to_gguf.py` + `conversion/base.py`@6d3568e31d: with `--outtype auto` FP8 weights are **kept** as
  F8_E4M3 only for tensors whose scale has one element (`if prod(scale.shape) != 1: continue`, in
  `_prepare_fp8_e4m3_tensors`); the scale is carried as a separate F32 tensor (graph-level multiply, per #28392
  description). Everything else (block-scaled FP8 such as 128x128 checkpoints, `weight_scale_inv`) is dequantized to
  BF16/F16 at conversion, or to Q8_0 with `--fp8-as-q8`. INFERRED consequence: the benefit of in-VRAM FP8 reaches only
  per-tensor-scaled checkpoints; common block-scaled HF FP8 models still become 16-bit (2 bytes/param) or Q8_0.
- READ: FP8 KV cache: `GGML_TYPE_F8_E4M3` is in `kv_cache_types` (`common/arg.cpp`), so `-ctk/-ctv f8_e4m3`.

### 1.2 HIP backend behaviour on gfx1100 (all READ in `ggml/src/ggml-cuda/`@6d3568e31d unless noted)
- `vendors/hip.h`: `FP8_AVAILABLE` is defined for HIP >= 6.3 (includes `<hip/hip_fp8.h>`), but
- `common.cuh` `ggml_cuda_f8_e4m3_to_fp32` and `ggml_cuda_fp32_to_f8_e4m3` take the **software** branch when
  `GGML_USE_HIP` is defined (the hardware branch is NVIDIA-only, `!defined(GGML_USE_HIP)`): decode uses `ldexpf`; encode uses
  `signbit/fabsf/frexpf/__float2int_rn` with range clamps. A hardware CDNA3 path exists only for the NVFP4 scale
  (`ggml_cuda_ue4m3_to_fp32`, gfx94x, e4m3fnuz).
- `fp8.cu`: the native FP8 matmul (`ggml_cuda_mul_mat_fp8`) is compiled only for non-HIP CUDA >= 11.8; on HIP the
  stub returns false.
- Decode (batch <= 8): `mmvf.cu` `mul_mat_vec_f<ggml_fp8_e4m3_t,...>` loads two bytes, converts each to FP32 in
  software (`mmvf_load_fp8x2` falls through to `ggml_cuda_cast<float>` per byte on HIP), FP32 `ggml_cuda_mad`
  against FP32 activations. `ggml_cuda_should_use_mmvf` returns `ne11 <= MMVF_MAX_BATCH_SIZE (8)` for F8 on all vendors.
  No packed-half, no dot2, no vectorized wide load beyond the 2-byte pair.
- Prefill (batch > 8): `ggml_cuda_mul_mat_cublas` sets `compute_type = F16` when fast FP16 exists for `F8_E4M3`
  and converts the whole weight to a temporary FP16/FP32 buffer (`convert.cu`, `convert_unary_cuda<ggml_fp8_e4m3_t>`)
  before calling hipBLAS. INFERRED: per-op temporary of 2 bytes/param for the layer being run (not the whole model),
  plus one extra read+write pass over the weights each time; same shape as the "dequant + cuBLAS" path whose cost ORT
  measured (section 4). The MMF/MMQ tile kernels have no F8 case that I found (UNKNOWN: not exhaustively grepped).
- KV cache: FP8 K/V live in the vec flash-attention kernels (`fattn-vec.cuh`, `fattn.cu` cases F8_E4M3 x
  {F16,BF16,Q4_0..Q8_0}); writes go through `cpy.cu` / SET_ROWS using `ggml_cuda_fp32_to_f8_e4m3` (software on HIP).
  READ: PR #32183 (merged 2026-11-13) had to whitelist the D=256 FP8 vec FA kernels on the gfx908 CI because they use 257
  VGPRs (2 spilled); "the register pressure itself is left to the AMD side". INFERRED: software FP8 decode raises
  register pressure; open PR #31802 (diabluz, 2026-11-06) fuses q8_0/q4_0/f8_e4m3 KV dequant into the MMA FA kernels (CUDA).
- Open PR #32288 (2026-11-14): refuse FP8 KV cache cleanly on backends without SET_ROWS F8 (Metal); HIP is not named.
- UNKNOWN: any measurement of F8 decode speed on gfx11 vs F16/Q8_0. No issue or PR found for "FP8 RDNA3" (searches:
  `F8_E4M3 HIP`, `fp8 RDNA3` returned nothing on 2026-11-15).

### 1.3 Vulkan backend
- READ: master has no F8 mention in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`@6d3568e31d. PR #29791 "vulkan: add fp8 and
  scaled matmul support" (0cc4m, open, updated 2026-11-14) adds `dequant_f8_e4m3.comp`, `mul_mat_vec`/`mul_mm`/`mul_mm_cm2`
  support, `mul_mat_scale.comp`, copy shaders. Body is two lines; contents not read in detail. UNKNOWN: whether it
  widens in registers for `mul_mm` or dequantizes to a temp; whether the coopmat path (RDNA3 has `VK_KHR_cooperative_matrix`
  on recent RADV) is used for F8.

### 1.4 Contributions
- READ: `CONTRIBUTING.md`: AI use allowed with disclosure; features start as an issue; bug fixes need a test;
  new ggml types carry extra criteria (already met by #27512). Kernel work inside `ggml-cuda` is routine
  (the F8 path was added by an NVIDIA engineer; the CI-fix #32183 by an outside contributor). Required checks:
  `test-backend-ops`, perplexity and `llama-bench` before/after. INFERRED: a HIP-specific F8 mmvf / widening kernel is
  an ordinary PR if framed as optimizing the existing type; plug-in points are `mmvf.cu` (`mmvf_load_fp8x2`,
  `mul_mat_vec_f` fp8 branch), `convert.cu` (the F8 unary conversion), `ggml-cuda.cu` (`ggml_cuda_mul_mat`
  dispatch near the call of `ggml_cuda_mul_mat_fp8`), `common.cuh` (`ggml_cuda_f8_e4m3_to_fp32`,
  `ggml_cuda_fp32_to_f8_e4m3`), `cpy.cu` (KV writes).

## 2. vLLM on ROCm (main@9bc16ae6a3)

- READ: `vllm/platforms/rocm.py`: `supports_fp8()` is `on_cdna() or on_rdna()` (gfx11 included, since the closed outside
  PR #31184 "Fix FP8 support detection on gfx11x", 2025-12, was not merged but master reached a similar result);
  `fp8_dtype()` is `float8_e4m3fnuz` only on gfx94x, otherwise `float8_e4m3fn`; supported quantization list includes
  `fp8`, `quark`, `compressed-tensors`, `awq`, `gptq` etc.
- READ: FP8 linear kernel candidates for ROCm: `vllm/model_executor/kernels/linear/__init__.py` `_POSSIBLE_FP8_KERNELS[ROCM]` =
  AiterHipbMM/AiterPreshuffled/AiterPerToken (AITER, CDNA-oriented), `ROCmFP8ScaledMMLinearKernel` (skinny GEMM
  `wvSplitKQ`, needs CDNA3+ or gfx12x), `PerTensorTorch`/`RowWiseTorch`/`ChannelWiseTorch` (`scaled_mm/pytorch.py`,
  gated by `_rocm_torch_fp8_scaled_mm_supported()` = gfx942/950/12x/1250). No gfx11 entry. `MarlinFP8ScaledMMLinearKernel`
  (weight-only W8A16) is `is_supported` only on CUDA (`scaled_mm/marlin.py`: "requires CUDA"). `TritonFP8ScaledMMLinearKernel`
  exists but is not in the ROCm list. INFERRED: loading an FP8 checkpoint or `--quantization fp8` on gfx1100 ends in
  `ValueError: no kernel can implement ...` from `choose_scaled_mm_linear_kernel` (the function raises when every
  candidate fails). I did not run it. UNKNOWN: whether some Fp8 path (`Fp8Config` online) special-cases gfx11.
- READ: FP8 KV cache: `v1/attention/backends/triton_attn.py` lists `fp8`, `fp8_e4m3`, `fp8_e5m2` and rejects FP8 only
  `if current_platform.is_cuda()` and capability < 8.9; the ROCm branch sets `supports_quant_query_input = not (on_gfx90a() or on_gfx11())`
  (query stays 16-bit on gfx11). `rocm.py` prefers `TRITON_ATTN` over `ROCM_ATTN` on gfx1100/gfx1201 unless `fp8_e5m2`.
  The cache write kernels use `__hip_cvt_float_to_fp8` in `csrc/quantization/w8a8/fp8/amd/quant_utils.cuh`. INFERRED:
  FP8 KV cache is accepted on gfx1100 and runs via Triton with in-kernel upcast and `hip_fp8.h` conversion, which is
  software on gfx11 (no FP8 instruction). UNKNOWN: measured speed or correctness on gfx1100.
- READ: AITER README (ROCm/aiter): gfx1100 is "Experimental"; on RDNA "Triton and most FlyDSL kernels run, as do
  most HIP kernels (norm, RoPE, quant, activation, plus some GEMM/attention). Most CK and ASM kernels are CDNA-only."
  https://github.com/ROCm/aiter (main@95d2112848, 2026-11-15).
- Maintainers: AMD ROCm team (AITER, `csrc/rocm/skinny_gemms*.cu`), many outside contributors. Open: PR #55996
  (2026-09, RDNA4 FlyDSL SplitKV with Triton fallback). vLLM accepts outside kernel PRs; the one gfx11 FP8 outside PR
  I found was closed unmerged (#31184, 6 lines, no testing evidence). `docs/getting_started/installation/gpu.rocm.inc.md`
  lists gfx1100/1101 as supported GPUs.
- Plug-in point (READ for structure): a new `FP8ScaledMMLinearKernel` subclass in `vllm/model_executor/kernels/linear/scaled_mm/`
  (`is_supported`, `can_implement`, `process_weights_after_loading`, `apply_weights` / `apply_scaled_mm`) and an entry in
  `_POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM]`. The Marlin FP8 class is the template (it repacks weights at load and does
  W8A16). There is also a second precedent on ROCm: Triton W4A16 (`kernels/linear/mixed_precision/triton_w4a16.py`).

## 3. PyTorch on ROCm (main@48f1cea6cf)
- READ: `torch.float8_e4m3fn` etc. are storage dtypes with software conversion in c10 (`c10/util/Float8_e4m3fn.h`);
  INFERRED: `.to(torch.float16)` on gfx11 works on the generic elementwise path (I did not run it).
- READ: `_scaled_mm` needs the arch allow-list in section 0; gfx1100 absent. hipBLASLt gating is the underlying reason (INFERRED).
- UNKNOWN: Inductor/Triton FP8 `tl.dot` on gfx11 (web search found no source; Triton's `float8e4nv` tests are
  NVIDIA-gated). INFERRED: casting to fp16 inside a Triton kernel then `tl.dot` is the viable route (SGLang does the
  analogue for int4, section 4).
- Contributions: PyTorch accepts outside PRs but ROCm kernels there are owned by AMD; an FP8 W8A16 op is better placed in
  vLLM/SGLang or a standalone extension than in core PyTorch (INFERRED).

## 4. SGLang (main@2e55a173a7), ONNX Runtime, MLC-LLM, ExLlamaV3, AMD stack

**SGLang.** READ docs (`docs/docs/hardware-platforms/amd_gpu.mdx`): FP8 "works via Aiter or Triton" (listed for
MI300X/MI325X/MI350X); a separate Strix Halo/gfx1151 (RDNA3.5, same WMMA ISA family as gfx1100) image exists.
READ `python/sglang/kernels/ops/quantization/w4a16_prefill_gfx1151.py`: a Triton W4A16 prefill GEMM for the gfx115x WMMA
that widens int4 to bf16 by OR-ing a magic exponent (`| 0x43004300`) and uses K=16 `tl.dot` calls; the decode GEMV
`w4a16_skinny_triton.py` is tuned on Radeon 8060S. Added by hubertlu-tw in commit `92b1d7be` (2026-11-07, AMD-side
author per the commit title tags; employer UNKNOWN). Claim in file: prefill 1.12-1.34x over their skinny GEMM from 256
rows. This is the best in-tree proof that a weight-only low-bit widen-then-WMMA kernel is accepted for gfx11-family.
FP8 on gfx11: UNKNOWN; `sg_fp8.py` (`srt/layers/quantization/fp8.py`) branches on gfx95/gfx1250 only; no gfx11 FP8 branch found.
Contribution surface: kernels live under `python/sglang/kernels/ops/quantization/`, gated by `use_gfx115_kernels()` in
`srt/utils/device.py` (opt-out env var `SGLANG_DISABLE_GFX115_KERNELS`); that gate is gfx115x only, not gfx1100 (READ).

**ONNX Runtime.** READ: the ROCm EP was removed (`onnxruntime/core/providers/` has no `rocm` dir on main@1f53982117;
PR #26712 "[ROCM] remove more rocm ep code", merged 2025-12-03). MIGraphX EP remains; I found no FP8 code in
`providers/migraphx` (search returned none; UNKNOWN for graph-level FP8 handled inside MIGraphX itself). Relevant
precedent: `com.microsoft::MatMulBlockQuantizedFp8Weight` exists for CUDA and WebGPU:
- WebGPU doc `docs/contrib_ops/webgpu/matmul_block_quantized_fp8_weight.md`: FP16 activations, E4M3FN weights with FP32 block
  scales, "FP8 storage with FP16 widening, not native FP8 matrix multiplication"; prefill uses 16x16x16 FP16 subgroup
  matrices when available. INFERRED: on Linux the WebGPU EP runs on Vulkan (Dawn), so it could run on a 7900 XT; UNKNOWN whether
  the AMD Vulkan driver exposes the needed subgroup-matrix feature. A free baseline to test.
- CUDA doc `docs/contrib_ops/cuda/matmul_block_scaled_fp8.md` (+ `_experiments.md`): works on any SM80+, weights
  dequantized to the activation type then cuBLAS, with a fused GEMV for M <= 8 (tensor-core mma variant; env `ORT_FP8_GEMV_MMA`).
  READ numbers on RTX 5060 Ti (448 GB/s): decode GEMV M=1 0.076 ms vs dequant+cuBLAS 0.396 ms at M=16..64 (4096x4096) and
  the dequant step only reached ~125 GB/s (28% of peak) before a 1.5x speedup by vectorizing.
  Lesson: the dequant-then-GEMM path is dominated by the dequant pass for M in 16..128.

**MLC-LLM / TVM.** READ: FP8 support (`python/mlc_llm/quantization/per_tensor_quantization.py`, `block_scale_quantization.py`) targets
CUTLASS and Triton kernels (`op/cutlass.py`, `op/triton.py`); there is a generic `convert_uint_packed_fp8_to_float`
dequant path for storage dtype. UNKNOWN: whether any of it compiles for ROCm gfx11; no gfx11 mention found. Recent commits
are dependency bumps (Windows, 2026-11-14). I would not target it first.

**ExLlamaV3** (turboderp). READ: has a ROCm/RDNA path: `exllamav3_ext/rocm/rdna_wmma.cuh`, `wmma_gemm.cuh`,
`quant/exl3_gemv_tiles_rdna.cuh` (decode GEMV; header says VALU, not DRAM, was the binding unit; uses `fdot2`, `v_pk_*`
ops), "generic gfx11 and gfx12 family" support (commit 2026-10-06). FP8 there only appears as dequant of FP8 *input*
checkpoints in `modules/linear.py` (`apply_fp8_scales_`), i.e. conversion to its own EXL3 format (INFERRED; I did not
read it fully). ExLlamaV2 last pushed 2026-03-04 (stale). Useful as proof that a decode GEMV on gfx11 can be ALU-bound
once weights are 3-8 bits: the same risk applies to an FP8 widen kernel.

**AMD stack.** READ: AITER FP8 linears assume CDNA (README above). hipBLASLt library lists FP8 (hardware gate decides).
Composable Kernel (develop@c9d63a48e0, 2026-11-13): UNKNOWN (search rate-limited). AMD Quark: produces FP8 (OCP e4m3)
checkpoints for vLLM/SGLang on Instinct per AMD's tutorial; INFERRED it is runtime-agnostic and does not itself give
gfx11 an FP8 kernel (web search found no gfx1100 statement).

## 5. Closest precedents for "FP8 storage, FP16 compute"

1. **feather-ops** (woct0rdho, MIT; https://github.com/woct0rdho/feather-ops@0e08b4e223, pushed 2026-11-05) and
   **ComfyUI-FeatherOps**. READ README: "On GPUs without native fp8 ... load fp8 data to LDS/VGPR then upcast to fp16
   before WMMA, rather than upcast the whole matrix to fp16"; `csrc/hip_kernel.cu` uses
   `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, B prepacked as `(K/16, N, 16)` raw bytes, FP32 accumulate, per-tensor scale
   in the epilogue, A fp16/bf16 (bf16 widened to fp16 at staging). Tuned for **E5M2** because E5M2 -> FP16 is a single
   `v_perm` (byte is the FP16 high half); E4M3 "supported" but decode costs 7.5% more (their notes: 42.46 vs 39.27 TFLOPS,
   `hip_optimization_plan_prepacked_e5m2.md` N3). E4M3 decode in `hip_kernel.cu`: `((x & 0x007f007f) << 4) + 0x3c003c00` plus sign.
   Tested on Strix Halo (gfx1151, ~212 GB/s) for image/video diffusion in ComfyUI; README says "should also work with
   RDNA3 GPUs". Their measured wins: 1.65-2.16x over `torch.mm` on an already-dequantized bf16 weight at large M (diffusion
   shapes M in the thousands), and a single M=1 cell at 227 GB/s = DRAM roof of that machine. Not an LLM runtime
   integration; no decode-GEMV-for-LLM tuning for a 800 GB/s card; no KV cache. INFERRED: strongest prior art and
   a natural collaborator; not a competitor on LLM decode.
2. **FP8 Marlin (NVIDIA Ampere)**: vLLM PR #5975 (mgoin, merged 2024-07-03, https://github.com/vllm-project/vllm/pull/5975):
   weight-only W8A16 for GPUs without FP8, "up to 2x in memory-bound scenarios", gains largest on lower-bandwidth GPUs
   (A10, RTX 3090); based on the GPTQ-Marlin 8-bit kernel: FP8 bytes packed into int32, permuted into Marlin tile layout
   offline, 4 x FP8 -> 4 x FP16/BF16 with bit ops (`csrc/quantization/marlin/dequant.h`: `(q & 0x80008000) | ((q & 0x7F007F00) >> 1)`,
   then a half-multiply by 2^8 to rebias, and that multiply is folded into the scale in the `true` variant), `mma.m16n8k16` fp16
   with FP32 accumulate (`marlin_mma.h`), no MoE at that time, channelwise scales. Per-layer A100/A10 charts exist only as
   images in the PR (numbers not extractable here). Still in vLLM today for CUDA (`scaled_mm/marlin.py`, `mxfp8/marlin.py`).
3. **ORT FP8-weight ops** (CUDA, WebGPU): section 4.
4. **Petit** (causalflow-ai/petit-kernel, BSD-3, pushed 2026-09-05): FP16/BF16 x FP4 (NVFP4/MXFP4) kernels for AMD
   **CDNA2/3/4** (README lists MI200/300/350), widened to 16-bit before MFMA; used by SGLang ("petit_nvfp4"). Same idea, wrong ISA family for gfx11.
5. **SGLang gfx115x W4A16** (section 4): same idea for int4 on WMMA, in Triton.
6. No other community project found doing FP8 weights with FP16 compute on RDNA 2/3 for LLM inference (web searches
   2026-11-15; UNKNOWN beyond that). RDNA 2 has no WMMA; the fallback would be `v_dot2_f32_f16` (RDNA2 has it on gfx1030? UNKNOWN).

## 6. (1) Table: runtime x gfx1100 FP8 behaviour

| Runtime | Loads FP8 checkpoints on gfx11? how | Memory cost | FP8 KV cache on gfx11 | Evidence |
|---|---|---|---|---|
| llama.cpp HIP (master) | Yes if per-tensor-scaled: kept as `F8_E4M3` in GGUF; block-scaled FP8 is converted to BF16/F16 (or Q8_0) at conversion time. Decode via generic `mmvf` software-decode kernel; prefill via per-op FP16 temp + hipBLAS | 1 B/param (kept) + a layer-sized FP16 temp in prefill; 2 B/param otherwise | Yes, `-ctk/-ctv f8_e4m3`, vec FA kernels with software decode/encode | READ (1.1, 1.2); speed UNKNOWN |
| llama.cpp Vulkan | Not on master; PR #29791 open | n/a | Not on master | READ (1.3) |
| vLLM ROCm | INFERRED: refuses (no FP8 linear kernel accepts gfx11; Marlin FP8 is CUDA-only); the 16-bit/AWQ/GPTQ/Quark-int4 paths work | n/a | Accepted in code via Triton attention + `hip_fp8.h` software conversion; untested here | READ code, INFERRED outcome |
| PyTorch ROCm | Tensors store/convert (software); `_scaled_mm` not allowed on gfx1100 | 1 B/param while stored | n/a (library) | READ (0, 3) |
| SGLang ROCm | FP8 via Aiter/Triton documented for Instinct; gfx11 UNKNOWN. W4A16 Triton exists for gfx115x | n/a | UNKNOWN | READ docs, UNKNOWN |
| ONNX Runtime | ROCm EP removed; MIGraphX no FP8 code found; WebGPU EP has FP8-weight op (FP16 widening) possibly via Vulkan | 1 B/param + block scales (WebGPU) | UNKNOWN | READ (4); Vulkan feature UNKNOWN |
| MLC-LLM | FP8 code targets CUTLASS/Triton; ROCm gfx11 UNKNOWN | UNKNOWN | UNKNOWN | READ partial |
| ExLlamaV3 | No FP8 weights at runtime; FP8 checkpoints only as quantizer input | EXL3 bits | no FP8 cache found | READ partial |
| feather-ops / ComfyUI | Yes, FP16 x FP8(E5M2 tuned, E4M3 supported) GEMM, gfx1151 tested | 1 B/param | n/a (diffusion) | READ |

## 7. (2) Where our two pieces help, ranked

Pieces: **W** = FP8 weight-only, in-register widening to FP16 for WMMA/`v_dot2_f32_f16`; **R** = exact fast FP32 -> E4M3
rounding (7.7 / 11.3 cycles per value).

1. **W for decode GEMV and prefill in llama.cpp's HIP path** (`mmvf.cu` + F8 prefill dispatch). Best fit: the F8 type
   landed 2026-10/11, HIP has only generic software code (`ldexpf` per byte, FP32 FMAs, FP16 temp for prefill), the ggml-cuda
   owners include a HIP owner, and nobody has a PR (UNKNOWN beyond searches). Caveat (INFERRED): audience is per-tensor-scaled FP8
   checkpoints only (see 1.1), and Q8_0 has the same bytes with mature kernels.
2. **W as a vLLM ROCm FP8 W8A16 kernel for gfx11** (Triton first, HIP later). Today vLLM on gfx1100 has no FP8 weight path
   at all (INFERRED), the plug-in is a clean class (section 2) and the Marlin FP8 precedent proves the design. Cost:
   vLLM review bar, AMD ownership, repack/scale layouts, torch.compile and CUDA-graph compatibility.
3. **W as E4M3 + decode-GEMV extension of feather-ops** (partner). It has the WMMA prefill kernel and E4M3 support,
   not LLM decode or a 7900 XT bandwidth regime; our measured E4M3 decode bit-tricks and the "defer the 2^8 rebias into the
   scale" idea (INFERRED: `(x & 0x7F) << 7` puts E4M3 into an FP16 whose value is value*2^-8; exact for normals and
   subnormals; multiply back in the FP32 epilogue, so decode is perm + and + shift) could cut its 7.5% E4M3 penalty.
4. **R in FP8 KV-cache writes** (llama.cpp `cpy.cu`/SET_ROWS software encoder; vLLM `reshape_and_cache` via
   `__hip_cvt_float_to_fp8`). Real but small: per decoded token the number of values quantized is
   2 x layers x kv_heads x head_dim (about 65k for a 32-layer, 8-kv-head, 128-dim model; INFERRED arithmetic), tiny next to the weights
   GEMV. Matters more in prefill bursts and for bit-exactness against other backends. The KV *read* side (FP8 -> FP16 widening in
   attention) is a W-type win with the D=256 register-spill evidence behind it (1.2).
5. **R for activation quantization (W8A8-style) on gfx11**: no FP8 instruction to consume it; IU8 WMMA already exists, so
   INT8 activations would beat FP8-emulated ones. Low value.
6. **Already done by someone:** FP8-weight widening for WMMA on gfx11-family (feather-ops, diffusion), FP4/FP8 weight widening on CDNA (Petit), FP8 weights
   on CUDA Ampere (FP8 Marlin), FP8 weights with FP16 subgroup matrices in WebGPU (ORT). Not found for LLM runtimes on gfx1100.

## 8. (3) Best first prototype and how to judge it

**Prototype:** a standalone HIP micro-benchmark (not a runtime patch), `fp8w_gemv` + `fp8w_gemm`, in this repo's
`carry-over` style, then port to llama.cpp `mmvf.cu` if it wins.

Kernel shape:
- Decode GEMV, batch 1..8: one wave32 per output row (or per 2-4 rows), 16-byte (`uint4`) loads of FP8 weights, widen 4 bytes to
  2 packed `half2` with `v_perm`/and/shift (E4M3, deferred 2^8 as above) or LUT-free E5M2 where the checkpoint allows,
  `v_dot2_f32_f16` against FP16 activations in registers, FP32 accumulate, wave reduce, scale in the epilogue (per-tensor first;
  per-row or 128-block next).
- Prefill GEMM, M = 64..4096: WMMA `v_wmma_f32_16x16x16_f16_w32`, B prepacked `(K/16, N, 16)` bytes (feather-ops layout, MIT),
  decode B in registers between LDS read and WMMA; tile 64-128 rows.

Measure on the RX 7900 XT (headless, memory-capped, under the project's GPU lock; nothing run in this task):
1. Roofline anchors: device copy bandwidth (GB/s), and the FP16 GEMV of the same kernel with 2-byte weights.
2. Decode: M = 1, 2, 4, 8; N x K = 4096x4096, 4096x14336, 14336x4096, 152064x4096 (LM head); report us, GB/s of weight bytes,
   and ns/token-equivalent. Compare: (a) FP16 weights (same kernel), (b) llama.cpp F8 `mmvf` (via `test-backend-ops perf` or
   `llama-bench` on a small per-tensor FP8 GGUF), (c) llama.cpp Q8_0 and F16, (d) our FP8 kernel.
3. Prefill: M = 64, 256, 1024, 4096 at the same N x K; TFLOPS, weight GB/s; compare: FP8-widen WMMA, FP16 WMMA, llama.cpp F8
   prefill (dequant + hipBLAS), feather-ops as-is (E5M2 and E4M3).
4. ALU headroom: VALU instructions per weight from the ISA dump (INFERRED budget: roughly 16 lane-instructions per weight
   at 800 GB/s on 84 CUs x 64 lanes x ~2.4 GHz; these are vendor-spec numbers, to be replaced with measured copy bandwidth and clock).
5. VRAM: resident model bytes and peak temp bytes for a 14B-class model (FP8 vs FP16) at the same context.
6. Accuracy: output vs an FP64/FP32 reference of dequantized weights (relative L2, max abs), identical results across 3 runs.
7. Quality of the widening itself: all 256 E4M3 codes -> FP16 exact, NaN code policy equals llama.cpp (`0x7F` -> NaN, saturating encode).

**What would make it not worth doing:**
- llama.cpp F8 decode on gfx1100 already reaches >= ~85% of copy bandwidth (then the software decode is not the limit; just check once).
- Our kernel gains < ~15% over F16 at batch 1 after the 2x byte saving, i.e. it is ALU-bound like the EXL3 decode.
- Q8_0 (same bytes, mature MMVQ/MMQ kernels) matches FP8 on both speed and quality for the target models (INFERRED likelihood: high);
  then the only remaining value is "exact checkpoint fidelity", a thin audience. Decide before porting.
- The target audience (per-tensor-scaled FP8 checkpoints) has no model people actually want to run on a 20 GB card (UNKNOWN; check which HF
  FP8 models use per-tensor scale; many use 128-block scales that llama.cpp dequantizes today).
- vLLM or llama.cpp maintainers already have an open PR for the same kernel by the time we port (re-check `gh pr list` first).

## 9. Open items I could not settle
- Composable Kernel / hipBLASLt gfx11 FP8 kernels; Quark on gfx11 (UNKNOWN).
- Actual error text and behavior of vLLM FP8 on gfx1100 (INFERRED only).
- PR #29791's Vulkan shaders (widening vs temp buffer) and whether RADV exposes coopmat for the F8 path.
- Measured F8 decode/prefill speed in llama.cpp HIP on gfx1100 (no source; first thing to measure).
