# RDNA 3 kernels and drivers: matrix ops under lane masks, D3D12 under Proton, FP8 emulation

These are the hardware, compiler and driver findings from a second workload on the same RX 7900 XT: GPU kernels for
the convolution and transformer layers of an image-upscaling network, written in HIP (Linux) and in D3D12 compute
(HLSL, run under Proton). The first part is about things that are not FP8 arithmetic and that I did not find
described elsewhere. The FP8 arithmetic itself (rounding, widening, the WMMA sums) is in the other files here and is not
repeated.

Labels: MEASURED = observed on the card in a run. READ = from documentation or source code, with a numbered source
in the list at the end. INFERRED = reasoned from the two. UNKNOWN = not found out.

| Item | Value |
|---|---|
| GPU | Radeon RX 7900 XT (gfx1100), Linux |
| HIP | ROCm 7.1.1, clang 20 |
| Vulkan driver | RADV, Mesa 27.0.0-devel, `VK_KHR_cooperative_matrix` revision 2 |
| D3D12 | vkd3d-proton `5b1089c8` built with `enable_extended_emulation`, dxil-spirv `114e664` plus a local raw-fragment change (not upstream), native DXC 1.10 for HLSL to DXIL |
| Wine | Wine 11.21 Staging for headless runs, GE-Proton 11-11 for the adapter-id test |

## Short answer

New, not about FP8 arithmetic:

1. **A matrix instruction can end up under a lane mask on gfx11, and then the results are wrong.** In one kernel the
   compiler moved a `v_wmma` chain into the branch of a per-lane store. On partly valid tiles the output had NaN codes
   or wrong values. A CPU emulator of the kernels cannot show this, because it runs every lane. Fixed in the source,
   proven by a test that is red on the old code, and a scan of the disassembly now finds every such site
   (section 1).
2. **`v_permlanex16` reads the register of a lane that is switched off, when the fetch-inactive bit is set.** Measured
   with five mask shapes; this is why the exchange is safe in most masked code (section 2).
3. **F16 matrix ops through D3D12 under Proton work on RDNA 3 and equal the HIP instruction bit for bit.** The lane
   layout the raw 8-element fragment needs is not the RDNA 4 one, and half of the matrix op's K range is lost unless
   you know it. Several driver traps are listed (section 3).
4. **Headless Wine needs two settings that are easy to miss:** Wine's null graphics driver, and no `XDG_RUNTIME_DIR` that
   points at a real session (section 4).
5. **Under Wine and Proton `D3DKMTQueryAdapterInfo` does not answer the physical-adapter-ids query.** An application that
   identifies the GPU through it must fall back (section 5).

FP8-related:

6. **Unchanged RDNA 4 FP8 shaders run on RDNA 3 if dxil-spirv emulates FP8 on raw fragments.** Output equals the native
   gfx11 path on every probe. The cost is almost all FP8 decode: a prototype with shift widening and one full-size
   matrix op took a synthetic whole-image pass from about 32 ms to about 11 ms (section 6).
7. **The same HIP kernels have 14 times (whole code object) to 48 times (one block kernel) the vector instructions on
   gfx11 that they have on gfx12**, and widening F16 weights once per workgroup halved or better the time of the
   transformer blocks (section 7).

## 1. A matrix instruction under a lane mask (gfx11)  [NEW]

**What looked wrong.** A kernel that expands a feature map (a 16-token tile per wave) wrote NaN codes, or wrong
values, for valid tokens, but only when a tile was partly valid: layers of 60, 84, 120 or 140 tokens, not 64 or 128.
Everything else passed on the card, and the CPU emulator of the same kernel passed everywhere.

**What we found.** The store was `if (token < ntok)`. The compiler put the whole `v_wmma` chain that produces the value
into that branch, because the store was its only user. In the disassembly the `v_wmma` chain sat after
`s_and_saveexec_b32`, whose mask came from a per-lane compare on the token index. The same move made the kernel spill
107 registers (the merge kernel 101).

**Why the emulator missed it.** The emulator models the kernels lane by lane and always runs all 32 lanes of a wave.
It has no EXEC mask, so the situation cannot arise in it.

**What the documentation says.** Both ISA guides define the matrix instruction with the whole wave: the pseudocode
saves EXEC, sets it to all ones, does the matrix multiply, and restores EXEC [1][2]. So on paper the hardware uses
every lane's operand registers whatever EXEC is. Our observation does not contradict that.

- INFERRED mechanism: the code that fills the operand registers (loads, half-wave exchanges) moved under the same mask,
  so the registers of the switched-off lanes were never written and held old values. The matrix instruction used them.
  In the disassembly above, the operands are loaded from scratch inside the masked region.
- I did not separate this from a possible hardware effect. The fix removes the mask, so it works in either case.

**The fixes (source level, gfx11 only).**

- Stores without a branch. A lane past the last token repeats the work of the last token and stores the same bytes to
  the same place. There is nothing left to move the matrix chain into. Results are unchanged (bit-identical in the
  emulator). The kernels also stopped spilling (0 spilled registers; merge 0.639 to 0.422 ms and expand
  0.688 to 0.418 ms at 1706x960).
- A wave-uniform loop bound. A second family of kernels had `if (win0 + i >= nwin) break;` in a window loop. `win0` is
  the wave's first window, but to the compiler it is a per-lane value, so the exit became a lane-divergent branch
  around every `v_wmma` and `v_permlanex16` of the loop. Reading `win0` through `readfirstlane` makes it scalar. The
  break is then a scalar branch and the whole loop runs with EXEC full.

**How it was proven.**

- Test red on the old code: a card test runs each kernel alone with partly valid tiles against the emulator. It fails
  with the branch put back and passes with the fix.
- The second family could not be shown wrong on the card. At run time its mask is all or nothing, so every card case
  passed before and after. For it the test that is red on the old code is the scan below (4 kernels flagged on the old
  build, none on the fixed one). A card test for partly filled workgroups (1 to 17 windows) is green on both builds. I
  checked that this card test can fail by making the window count off by one.

**The scan.** A script disassembles a code object and runs a forward data-flow over the control-flow graph of each
function. EXEC becomes partial at `s_and_saveexec`, at writes to EXEC by `s_and`, `s_and_not1`, `s_xor`, and at
`v_cmpx`. It is restored by `s_or exec` or `s_mov exec` from a register that was saved from EXEC, and after a
divergent loop. Paths merge as "partial if any path is partial". It lists every `v_wmma` and `v_permlanex16` that may
run with a partial mask. The test for it has 15 hand-written functions, each rule with a look-alike that must not be
flagged, and a scan of every kernel the runner launches.

What the scan found beyond the two fixes, each checked by hand:

| Flagged | Verdict |
|---|---|
| 117 `v_permlanex16` in 39 post-processing kernels (3 each, no matrix op) | Harmless, see section 2. Pinned as known exceptions with exact counts. |
| Fused merge and expand kernels (216, 300 and 32 sites per function) | The mask comes from the loop entry test `tid < 32 * tiles`. The bound is a multiple of 32, so the mask is full or empty at run time. Making it scalar added 10 to 25 spilled registers. Pinned with exact counts instead. |
| One persistent kernel | Still carries the pattern and is not fixed. It also spills 143 registers for a reason I did not find, and it is slower than the plain kernels, so it is not used. |

Pinning means a new site, a missing one, or a changed count fails the test. I checked that it goes red when a count
is changed by one, and when a pinned site is no longer flagged.

**RDNA 4 (observation only).** The matrix-op-inside-a-lane-mask pattern is also in the gfx12 code objects built from the
same sources, which the fix does not touch (it is gfx11 only). In the expand kernel, the first `v_wmma_f32_16x16x16_fp8_fp8`
is in an `s_and_saveexec` region on `v_cmp_gt_i32 s0, s1, v109`, the same per-lane token compare. The RDNA 4 guide
defines the instruction with EXEC forced to all ones as well [2]. UNKNOWN whether it misbehaves on RDNA 4: I have no
RDNA 4 card here and no run was made.

## 2. `v_permlanex16` under a lane mask  [NEW]

The half-wave exchange on gfx11 is one `v_permlanex16_b32` (lane `l` gets lane `l ^ 16`). The guide's pseudocode copies
the source register of all lanes first, and writes only the lanes with EXEC set [1]. Its text says the instruction's
OPSEL[0] is the fetch-inactive (FI) bit, and its example notes "FI bit needed for lanes 15 and 31" [1]. The LLVM
intrinsic takes `fi` and `bound_control` as constant arguments [3].

**Test.** A probe calls the exchange (with FI set) in code that only some lanes execute: lanes 0 to 15 only, 16 to 31
only, odd lanes only, every fourth lane, and a data-dependent half. MEASURED: in all five shapes every executing lane
got the value of lane `l ^ 16`, also when that lane was switched off, and the lanes that were off were not touched.
0 wrong values.

**Consequence.** An exchange inside a masked branch is safe when the partner's value was computed by all lanes before
the branch. The 117 flagged sites in section 1 are of this kind (a sum made by every lane, then exchanged inside the
store's branch). Proof they are right: 21 card tests of the post kernels, bit-identical to the variant that has no
exchange. They go red (191 failures) when the exchange returns its own value, and green again when restored.

UNKNOWN: the behaviour with FI clear. The probe and the kernels always set it.

## 3. F16 matrix ops through D3D12 under Proton  [NEW]

**Question.** Can D3D12 compute shaders use RDNA 3's matrix unit through vkd3d-proton, dxil-spirv and RADV? We were
checking before writing any kernel for it. All of this was run headless (section 4).

**What works, MEASURED.**

- A raw-fragment F16 multiply-add (F16 inputs, F32 accumulate) compiles and runs. It equals the HIP
  `v_wmma_f32_16x16x16_f16` bit for bit: 1.000000 of 65,536 values on small mixed-sign integers, N(0,1) with random C,
  and wide-exponent data. On the mixed-sign integers both differ from the exact result in 44 % of values. So the
  inexact sums from `matrix-arithmetic.md` come out of D3D12 too.
- K = 16 as two chained K = 8 ops (K 0..7, then 8..15 into the same accumulator) equals the single HIP K = 16
  instruction bit for bit on four data sets, maximum difference 0.
- A half-wave exchange written as `WaveReadLaneAt(v, lane ^ 16)` compiles to one `v_permlanex16_b32` in ACO.

**What the layout is, MEASURED.** A raw fragment is 4 dwords, 8 halves per lane, for A and B. Probes with one nonzero
element at a time gave:

| Operand | Rule |
|---|---|
| K range | Only the 8 fragment elements exist; element `e` is K index `e`. K 8..15 are zero-filled. The result equals the K = 8..15 product alone, not the full 16. |
| Accumulator | Element `i` of lane `l` is row `2i + (l >> 4)`, column `l & 15`. (The RDNA 4 kernels use row `8h + i`.) |
| A | Output row `r` reads its A data from lane `16*(r & 1) + r`. Lanes 1, 3, .., 15 and 16, 18, .., 30 are never read. |
| B | Output lane `l` reads its B column from lane `l` itself. Lanes `l` and `l ^ 16` must both hold column `l & 15`. |

An RDNA 4 shaped fragment (lane = row, half = K block) therefore gives even rows K 0..7 and odd rows K 8..15. The fix
is to fill both halves alike. The ISA guide says lanes 0 to 15 "are replicated into lanes 16-31" [1], which is the
rule to follow. The instruction is more than that: the card reads lanes 16 to 31 for the odd output rows. The HIP side
measured the same with raw registers (the lanes 16-31 of the result take A and B from lanes 16-31). Because each half
reads its own lanes, the two halves can carry different B matrices (INFERRED, not run).

A full K = 16 step costs two ops through this channel, half the card's matrix rate (the HIP form does it in one).

**What fails, MEASURED unless marked.**

| What | What happened |
|---|---|
| Any FP8 matrix op in a shader | Pipeline creation fails with `0x80070057` and the vkd3d log says "Failed to compile shader, vkd3d result -3". It kills the whole shader, also when it mixes F16 and FP8. Without native FP8 the raw-fragment path refuses the shader (from reading our dxil-spirv tree; not upstream, so no source is cited). |
| Cooperative matrices on gfx11 | Not exposed in the default vkd3d-proton build. The option `enable_extended_emulation` defaults to false; with it, the 8-bit types are taken from `UINT8` cooperative-matrix entries when there is no native FP8. The source comment says "Don't expose this by default in official builds" [4]. Without it the device logs "Missing sufficient features to expose WMMA." [4]. Side effect: `SupportsAGSExtension` for the FP8 matrix extension then returns true without native FP8, so software that asks for it will try the emulation [4]. |
| Some FP8 shaders | Not always refused: 5 of 46 RDNA 4 DXIL files page-faulted the driver at pipeline creation (23 created, 17 failed, 5 crashed) with the FP8 emulation of section 6 off. With it on, all are created and the crashes are gone. INFERRED: a failed raw-fragment build left an invalid shader; not investigated. |
| `DXIL_SPIRV_CONFIG=wmma_rdna3_workaround` | In a shader that uses WMMA it adds the execution mode `RoundingModeRTZ` for 16-bit floats, and the source comment says the hack "only makes sense on RDNA3" [5]. Our kernels need round-to-nearest, so it stays off. |
| `f32tof16` and `(half)` | In shaders containing a matrix op: `f32tof16` is round-toward-zero (0.9844 match with RTZ, 0.4883 with RNE on 256 edge values) and a `(half)` cast is round-to-nearest-even (1.0000 with RNE). The convert operation of AMD's AGS shader extension (opcode 0x32) is RNE as well. Code that assumes both round alike is wrong on gfx11. |
| Pipeline cache after a dxil-spirv rebuild | The shader cache is not keyed by the dxil-spirv build. After a rebuild, an old cache gave stale pipelines. Every run used a fresh cache path. |

On the `f32tof16` result: a Mesa rule rewrites vkd3d-proton's DXBC f32-to-f16 sequence into a round-toward-zero pack,
and its comment says that conversion "implements a float conversion using PackHalf2x16" [6]. That is the DXBC path. I
did not check the DXIL path in source, so a link to our result is INFERRED. The numbers above are what the card does.

**How the layout was found.** First a probe with a single nonzero A element, then a single nonzero B element, stepping
through lanes and K positions, and a model that turns a matrix into fragments. The model was then checked against
random data in the bit-for-bit comparison against HIP. A wrong model would not have matched HIP at 1.000000.

## 4. Headless Wine  [NEW]

The D3D12 probes had to run with nobody watching and without opening windows on the desktop. Two things were not
obvious.

1. **A null graphics driver is needed.** In a prefix with no display and the normal driver, `D3D12CreateDevice` failed
   with `0x887a0004`. Setting `[Software\Wine\Drivers] "Graphics"="null"` in the prefix made device creation, a compute
   pipeline and a dispatch work. MEASURED.
2. **`XDG_RUNTIME_DIR` can open real windows.** A probe crashed and Wine opened a debugger "Program Error" window on
   the real desktop, although `DISPLAY` and `WAYLAND_DISPLAY` were unset. `XDG_RUNTIME_DIR` still pointed at the user
   session. INFERRED mechanism: the Wayland client library connects to `wayland-0` inside `XDG_RUNTIME_DIR` when no name
   is given [7], and Wine's Wayland driver found the compositor's socket. The fix: a scratch, empty `XDG_RUNTIME_DIR`,
   the null graphics driver, the debugger disabled (`winedbg.exe=d` override) and `ShowCrashDialog=0` in the prefix.
   A tool that needs the real runtime directory (`systemd-run --user`) gets it only in its own client process, outside
   Wine.

   Proof: a program that writes through a null pointer, run with a once-per-second check of every process that held the
   prefix. No process saw the real runtime directory, none held a Wayland or X11 socket, and no window appeared.

## 5. Wine and Proton: the physical-adapter-ids query is not implemented  [NEW]

Some Windows software identifies the GPU with `D3DKMTQueryAdapterInfo` and the type
`KMTQAITYPE_PHYSICALADAPTERDEVICEIDS`, which Microsoft documents as returning "the device ID(s) of the physical
adapters" in a `D3DKMT_QUERY_DEVICE_IDS` structure, "supported starting with Windows 10 (WDDM 2.0)" [8][9].

**Test.** A small console program opens each adapter listed by DXGI with `D3DKMTOpenAdapterFromLuid`, and also the LUID
of an `ID3D12Device` created on it, and sends the query (type 31).

| Environment | Adapter | Open | Query |
|---|---|---|---|
| GE-Proton 11-11, normal desktop session | RX 7900 XT (RADV) | 0x0 | `0xC0000002` (`STATUS_NOT_IMPLEMENTED`), ids 0 |
| GE-Proton 11-11 | the CPU's integrated Radeon | 0x0 | `0xC0000002` |
| Wine 11.21 Staging, headless | Wine's placeholder adapter | 0x0 | `0xC0000002` |

The DXGI LUID and the D3D12 device LUID are the same, so the open works and only the query is missing. READ: in Wine's
`NtGdiDdDDIQueryAdapterInfo` only two types are handled (driver update status, driver version), and every other type
logs "not handled" and returns `STATUS_NOT_IMPLEMENTED` [10]. That is Wine 11.21 mainline; the Staging build and
GE-Proton gave the same status.

**Fallback.** Detect Wine and take the vendor and device ids from DXGI instead, only when the kernel query returns
`STATUS_NOT_IMPLEMENTED`. MEASURED with that fallback: the RX 7900 XT is identified as `1002:744c`. Not run: an RDNA 4
card under Proton (none in this machine), so the end-to-end result for that card is INFERRED from the code.

## 6. Running unchanged RDNA 4 FP8 shaders on RDNA 3  [FP8]

**Idea.** The raw-fragment support in dxil-spirv lets a shader pass A, B and accumulator dwords directly to the matrix
op. On RDNA 4 these dwords hold FP8. If dxil-spirv decodes the FP8 bytes to F16 and uses two chained F16 ops, the
RDNA 4 shaders should run unchanged on RDNA 3. The exact FP32-to-FP8 rounding of `e4m3-rounding.md` is used for the
conversion op. Everything here is a local change on top of that support; none of it is in upstream dxil-spirv.

**What it does.** Per K half, the lanes are shuffled (`OpGroupNonUniformShuffle`) to the gfx11 layout of section 3,
FP8 bytes are widened to F16 exactly, and two F16 multiply-adds run. The A rows are permuted so that the accumulator
element `i` is the same row on both architectures, which makes the accumulator need no shuffle.

**Result, MEASURED, synthetic data only.**

- The encoder kernel: output equals the native gfx11 HIP kernel byte for byte in 14 of 14 probe cases (including NaN
  and infinity inputs). Transformer blocks: byte-equal for the 96-channel block and the block with a fused expand. The
  64-channel block and its patch merge differ in a few values, each by one FP8 code: 3 of 30,720 on a small case, and
  0.007 % and 0.030 % on a case at the 1706x960 stage size.
- Unit probes: FP8 multiply-add mean relative error 1.4e-7; FP32 to FP8 conversion 0 wrong of 16,384 including every
  midpoint and its neighbours.
- All 46 pipelines are created (before: 23 created, 17 failed, 5 crashed).
- A real application run showed banding that I could not reproduce in the headless runs; its cause is UNKNOWN and
  nothing I measured points at the emulation.

**The trap that cost the most time: the driver folded the rounding away.** The shaders round to F16 on purpose by
converting F32 to F16 and back. The results were off by one FP8 code in 0.8 % of the encoder outputs. In the final
NIR (`RADV_DEBUG=shaders`) the loads fed the stores and the conversions were gone. Probe: 16,254 of 16,384 wrong before
the fix, 0 after. Three attempts: an explicit round-to-even mode (no effect), another widening instruction (no effect),
and `NoContraction` on the two conversions (fixed it). The third was a change I could see in the NIR, not a guess.
Mesa has a rule that removes `f2f32(f2fmp(a))` and a comment saying such conversions "should only be removed if the
instruction was generated by the precision lowering pass" [6]; its own `fquantize2f16` lowering marks the pair as exact
[6]. INFERRED: that rule (or one like it in the Mesa we ran, newer than the tag cited) is what fired. I did not identify
the exact rule. The consequence is real either way: correctness depends on the driver's compiler. A start-up rounding
probe is the safe guard. I did not search the other shaders for the same pattern.

**Cost anatomy, MEASURED (ACO disassembly, synthetic model, no real data).** The first version was 1.4x (encoder) to
1.85x (transformer stage) slower than the native HIP route. Counting instructions of the largest block kernel:

| Largest block kernel, static, per wave pass | Native HIP gfx11 | Emulation, first version | + shift widening and one K = 16 op |
|---|---|---|---|
| Instructions | n/a | 78.6 k | 34.3 k |
| Vector (VALU) instructions | 62.8 k | 62.6 k | 27.1 k |
| Matrix ops (`v_wmma`) | 312 | 736 | 368 |
| Spilled VGPRs | | 1140 | 86 |

- About 70 % of the instructions are the per-byte FP8 decode, about 13 issue slots per element. The shift form of the
  main README needs about 2.25.
- The kernels sit at the 256-VGPR limit (5 waves per SIMD), and four of five spill. The decoded operands stay live.
- Two K = 8 ops per K = 16 step: 736 ops against 312.
- Lane exchanges go through LDS (`ds_bpermute`), 768 to 2,432 per kernel.
- Not the cause: LDS size, global loads, barriers, launch gaps (together 0.005 ms for 247 barrier intervals).

**Speed-ups, MEASURED on a synthetic model, prototype numbers; the final patch is not finished.** Whole pass, input
1706x960 to output 2560x1440, minimum of 60 images, GPU shared with other work (the quiet baseline for the first version
is 31 to 32 ms):

| Version | Time | Output |
|---|---|---|
| First version | 31 to 32 ms quiet (36 to 37 ms with other GPU work) | reference |
| Shift widening (the main README's form, in dxil-spirv) | 16.3 to 16.6 ms | byte-equal to the first version on 3 images |
| Shift widening plus one full K = 16 op per step | 11.3 to 11.5 ms | byte-equal on 3 images |
| The same, plus NaN-code preservation by the word mask from the main README | 16.2 to 16.4 ms | byte-equal (no NaN code in the test inputs, so this path is untested) |
| Native HIP gfx11 route, for comparison | 17.9 ms | |

Keeping NaN codes costs about 4.9 ms, because the shift form turns the NaN codes into plus or minus 480. The test inputs
contain no NaN code. Open: only the final image was compared, not the stage buffers; whether one K = 16 op and two K = 8
ops stay identical beyond the data sets above (measured on four data sets in section 3; I found no documentation that promises it).
Also: a matrix floor of 3.5 to 4.7 ms for the whole pass, so every path is still 2.5 to 3 times above it.

## 7. Code size and F16 weights in the HIP kernels  [FP8]

**What was counted.** Vector instructions (VALU) per matrix instruction in the disassembly of the same kernel sources,
built for gfx1100 and for gfx1201 (static counts, not run time):

| Code object | gfx1100 | gfx1201 |
|---|---|---|
| Whole transformer code object (VALU / WMMA) | 335,098 / 1,456 = 230 | 23,006 / 1,936 = 12 |
| Largest block kernel, per window (WMMA 312 on both) | 62.8 k VALU | 1.3 k VALU |

INFERRED: much of the gfx11 count is the software FP8 path (widening, the quantizer, exchanges), and some is code that
is larger than it needs to be, as in the compiler pitfalls of the main README. The loop-level check from the main
README (`loop_isa.py`) is the next step before any claim.

**What helped most, MEASURED on synthetic weights, bit-identical to the old kernels.** Every matrix call widened both E4M3
operands to F16. The weights are used once per wave and window. Widening them once per workgroup, when the tile is
copied to LDS, cut the vector instructions (about 6,600 of 10,500 in the first block kernel). The step by step:

| Variant | Encoder-stage kernel, 1706x960 (ms) |
|---|---|
| Widen every call | 3.2 |
| Weights widened once per workgroup | 1.90 |
| Plus F16 activation operands, made once and straight from the FP32 values | 1.30 |

At 2560x1440 the last row is 2.82 against 6.8 ms. In isolation, the four transformer stages together went from 10.1 ms
to 4.5 ms with the last variant. Registers decided each step: F16 operands for the whole 128-channel block needed more
than 256 registers at first, and F16 operands for the whole 96-channel block spilled until only the larger half of the
block (the MLP) was changed. Each variant has a gate that records VGPRs, spills and waves per SIMD.

**One thing that did not carry over.** The exact FP32-to-E4M3 method of `e4m3-rounding.md` is faster, but in the block
kernels it took 30 to 35 more registers and made the largest kernel spill 27. The plain form stays there.

## How the work was done

The order mattered, and every step had a check that could fail.

1. **CPU emulator first.** The kernels were first run on the CPU in an emulator, lane by lane, against a float64
   reference. This gives correct reference numbers cheaply. It is also what hid the lane-mask problem (section 1).
2. **Card tests.** Each stage was run on the card against the same reference and against the emulator.
   Expected-failure tests recorded what was known to be wrong, so they turned into passing tests when fixed.
3. **Lane-mask scan** on the disassembly of every code object (section 1), after the first failure showed that card
   tests alone would not find a mask that is only wrong on some inputs.
4. **D3D12 probes** under headless Wine (sections 3 and 4): one small shader per question, compared bit for bit against
   HIP on the same card.
5. **FP8 emulation probe** (section 6) before any kernel was ported: one kernel first, then the 46-pipeline sweep, then
   a whole synthetic pass.
6. **Latency measurements** (sections 6 and 7): timestamp pairs per dispatch, ISA instruction counts per kernel,
   prototypes in a scratch copy of the source, interleaved runs.

Rules that mattered:

- **A test must be able to fail.** Each new rule was broken on purpose (the branch put back, the exchange removed, a
  count changed, the window count off by one), the test went red, and then the fix was restored and the test went
  green. A probe that shows a bug was also run where the bug cannot exist.
- **Compare against something independent:** a float64 reference, the CPU emulator, a CPU oracle for one output filter,
  and the other implementation (HIP against D3D12). Expected values come from the requirement, not from the output of
  the code.
- **One heavy job at a time, with a memory cap.** Two out-of-memory kills of a running worker, caused by other programs
  on the machine, led to a six-gigabyte cap per job and a lock for the GPU.
- **Headless runs only** (section 4), and a fresh shader cache path for every run.
- **Label the evidence.** Measured, read, inferred and unknown were kept apart in every note.

## Limits

- One GPU (RX 7900 XT, gfx1100). Nothing was run on RDNA 4, so every RDNA 4 statement here is about disassembly or the
  documentation.
- The matrix-op-under-mask mechanism (section 1) is inferred; the documentation says the instruction ignores EXEC.
- The FP8 emulation is a local change. It is not upstream and needs a patched dxil-spirv and the environment variable
  that enables it. The numbers in section 6 are from a prototype.
- Synthetic models and weights only. The timings depend on the GPU being quiet; where it was shared, minimums are given.

## Sources

Each was fetched on 2026-11-15 and the quoted text checked against it.

| # | Source | Used for |
|---|---|---|
| [1] | AMD, "RDNA3" Instruction Set Architecture Reference Guide, 15 August 2023. <https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0> | Section 7.9 (p. 73): "lanes 0-15 data are replicated into lanes 16-31". Section 16.10, `V_WMMA_F32_16X16X16_F16` (opcode 64, p. 358): pseudocode `saved_exec = EXEC; EXEC = 64'B(-1); ...; EXEC = saved_exec`. Section 16.12, `V_PERMLANEX16_B32` (opcode 604, pp. 424-425): OPSEL[0] is the FI bit, the example comment "FI bit needed for lanes 15 and 31", and the pseudocode that copies every lane's source and writes only lanes with EXEC set. |
| [2] | AMD, "RDNA4" Instruction Set Architecture: Reference Guide, 7 April 2025. <https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture> | Section 16.10, `V_WMMA_F32_16X16X16_F16` (opcode 64, pp. 399-400): the same EXEC pseudocode. |
| [3] | LLVM, `llvm/include/llvm/IR/IntrinsicsAMDGPU.td` at tag llvmorg-20.1.0, `int_amdgcn_permlanex16`: <https://github.com/llvm/llvm-project/blob/llvmorg-20.1.0/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L2660-L2665> | Line 2660: "llvm.amdgcn.permlanex16 <old> <src0> <src1> <src2> <fi> <bound_control>". (ROCm's clang 20 is AMD's fork; the upstream tag is the nearest public reference.) |
| [4] | vkd3d-proton at commit `5b1089c89dbc5cb9f662650feb60d2d3df3e55ca`: `meson_options.txt` (option `enable_extended_emulation`, default `false`); `libs/vkd3d/device.c`, function `vkd3d_supports_minimum_coopmat_caps`, lines 2380-2395 and 2417-2420; `libs/vkd3d/device_vkd3d_ext.c`, function `d3d12_device_vkd3d_ext_SupportsAGSExtension`, lines 659-686. <https://github.com/HansKristian-Work/vkd3d-proton/blob/5b1089c89dbc5cb9f662650feb60d2d3df3e55ca/libs/vkd3d/device.c#L2380-L2395> | The `#ifdef VKD3D_ENABLE_EXTENDED_EMULATION` block and its comment, the warning "Missing sufficient features to expose WMMA.", and the `WMMA_FP8` case that returns true under extended emulation without native FP8. |
| [5] | dxil-spirv at commit `114e664efca411c926380b2f3bd1930c1faeaf4d`: `dxil_converter.cpp`, lines 7782-7789 and the `DXIL_SPIRV_CONFIG` parsing at lines 10199-10203. <https://github.com/HansKristian-Work/dxil-spirv/blob/114e664efca411c926380b2f3bd1930c1faeaf4d/dxil_converter.cpp#L7782-L7789> | The `RoundingModeRTZ` execution mode for 16 bit when `wmma_rdna3_workaround` is set; the comment "This hack only makes sense on RDNA3." |
| [6] | Mesa at tag `mesa-25.3.0`, `src/compiler/nir/nir_opt_algebraic.py`: lines 1913-1915 (`f2f32(f2fmp(a))`), 3223-3227 (comment above `vkd3d_proton_packed_f2f16_rtz_lo`), 3410-3415 (`fquantize2f16` lowering with `!f2f32`, `!f2f16_rtne`). <https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.3.0/src/compiler/nir/nir_opt_algebraic.py#L1913-L1915> | The folding rule and its comment; the exact-conversion form; the vkd3d-proton conversion comment. The Mesa that was run is 27.0.0-devel, newer than this tag. |
| [7] | Wayland, "Appendix B. Client API", `wl_display_connect`. <https://wayland.freedesktop.org/docs/html/apb.html> | "If name is NULL, ... display "wayland-0" will be used", and a relative name "is opened relative to the XDG_RUNTIME_DIR directory". |
| [8] | Microsoft, "D3DKMTQueryAdapterInfo function (d3dkmthk.h)". <https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtqueryadapterinfo> | The function "retrieves adapter information from an adapter handle". |
| [9] | Microsoft, "KMTQUERYADAPTERINFOTYPE enumeration (d3dkmthk.h)". <https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/ne-d3dkmthk-_kmtqueryadapterinfotype> | `KMTQAITYPE_PHYSICALADAPTERDEVICEIDS`: "pPrivateDriverData points to a D3DKMT_QUERY_DEVICE_IDS structure that specifies the device ID(s) of the physical adapters. Supported starting with Windows 10 (WDDM 2.0)." |
| [10] | Wine at tag `wine-11.21`, `dlls/win32u/d3dkmt.c`, function `NtGdiDdDDIQueryAdapterInfo`, lines 810-847. <https://github.com/wine-mirror/wine/blob/wine-11.21/dlls/win32u/d3dkmt.c#L810-L847> | Only `KMTQAITYPE_CHECKDRIVERUPDATESTATUS` and `KMTQAITYPE_DRIVERVERSION` are handled; the default case is `FIXME( "type %d not handled.\n", ...)` and `return STATUS_NOT_IMPLEMENTED`. |
