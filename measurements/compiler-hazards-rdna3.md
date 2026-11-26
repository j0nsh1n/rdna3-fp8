# Do the two RDNA 4 compiler wrong-result cases also happen on RDNA 3 (gfx1100)?

## Summary

Both do. On the RX 7900 XT (gfx1100), clang 24 (ROCm 10.1 and 10.2) and the system clang 20 (ROCm 7.1.1) all leave the
same hole the RDNA 4 study found, and the matrix hardware is just as unforgiving.

1. **Call boundary: yes, same miss.** A non-inlined function that ends in a WMMA, called by code whose next
   instruction is a WMMA using that result as A, has no V_NOP between them in any of the four builds I made. In the
   plain C++ form the card gives 44 of 256 results wrong in every launch (8 launches x 3 jobs per build). A variant with
   the operands ordered by inline assembly gave 0, 0, then 44 in the remaining six launches of one job (ROCm 10.1, -O3);
   it gave 44 in every other launch. On one extra launch per job, the wrong results equal bit for bit what the forbidden
   inline assembly (no V_NOP) gives. The compiler does put the V_NOP in when everything is in one function.
2. **Per-lane condition: yes, with a gfx11 difference.** A WMMA under a per-lane `if`, an early `return` or a per-lane
   loop gives wrong results on the card whenever a switched-off lane held an operand the instruction needs. The safe
   forms (select padding, wave-uniform return, uniform loop) are right in every launch. One shape the RDNA 4 study saw
   fail, 16 valid lanes of 32, is right here. On gfx1100 each half of the wave computes its own output rows from its own
   lanes' copy of A and B, so when only lanes 0-15 store, their rows need only lanes 0-15's data (INFERRED from the
   layout section below; the n = 8, 20 and 31 failures fit it).
   Also: the system compiler moves a WMMA under the store's mask (as the README's old gfx11 note says), and that alone
   is harmless here, since the hardware ignores EXEC and all operands were loaded.

Both deliberate controls show the bug (the no-V_NOP break fails, the unsafe shapes fail), so the probes can see it.
Nothing here is a game or llama.cpp result: they are small probes, one wave per launch.

## Setup (MEASURED where marked)

- Card: RX 7900 XT, `lspci` 1002:744c, gfx1100. Kernel 7.3.4-202.nobara.fc44. System ROCm `rocm-hip-7.1.1-4.fc44`
  (clang 20.0.0.rocm). ROCm 10.1.0 and 10.2.0 under `~/.local/opt` (AMD clang 24.0.0git, two different commits).
- The 10.1 and 10.2 binaries ran against the runtime that `~/.local/opt/rocm` points to (10.1.0); the system binary against
  the system runtime. Code objects are fixed at build time, so the compiler is what differs.
- Card clock (READ from `pp_dpm_sclk`, not changed): before the final run `1107Mhz *`, after `1049Mhz *`. The probes are
  one wave and test correctness, not speed, so the clock does not enter the results. No sysfs writes, no sudo.
- Card profile (READ by the lead after the runs, 2026-11-26): LACT overclock, core up to 3000 MHz, -25 mV voltage offset,
  memory up to 1250 MHz; `power_dpm_force_performance_level` read `low`. The plain C++ call case gives the same count in every launch and the safe
  shapes and the inline-assembly control agree, so a voltage fault is not a likely cause (INFERRED). A stock-profile rerun
  was not made.
- Every job ran under `flock` on a GPU lock shared with the other jobs on the machine (`HAZ_GPU_LOCK`), `nice`, under 50 s. The lock was free every time; the kernel
  log showed no `ring timeout` or `GPU reset`.
- Inputs: integers 0 to 3, all terms non-negative. Reason (MEASURED): with mixed-sign inputs `v_wmma_f32_16x16x16_f16`
  was off by about 1e-6 on 107 of 256 results of a plain WMMA (4.00000095 against 4), so a signed probe shows "wrong"
  on a good run, which the README already says. With non-negative terms the CPU answer is exact and a plain WMMA
  matches it in 256 of 256.
- Sources: `src/hazards/haz3_hazard.hip` (case 1), `src/hazards/haz3_mask.hip` (case 2), `src/hazards/haz3_common.h`.
  Scans: copies of the RDNA 4 study's `haz_hazscan.py` and `haz_maskscan.py` in `scripts/hazards/`, unchanged (their
  listing syntax is the same on gfx11), with gfx1100 expectation files in `scripts/hazards/expect/`.

## What the hardware does (MEASURED unless marked)

- A and B: each half of the wave reads its own lanes, as `kernels-and-drivers.md` (lines 156-163) already measured:
  output row r reads A from lane 16 x (r & 1) + r, so the odd rows take A from the odd lanes 17-31, and output lane l
  reads its B column from lane l itself. These probes put different data in the two halves, so the rule shows in their
  results (my first stage-2 model assumed lanes 16-31 were ignored and failed on a good chain; fixed, noted in
  `expect.md`).
- D (f32): lane l, element e = row 2e + (l >> 4), column l & 15. D (f16, opsel 0): the low half of each of 8 registers;
  the high halves came back 0 here, where C's were 0.
- EXEC: with a partial EXEC the instruction still wrote every switched-off lane's D (128 of 128, 248 of 248 for the
  masks 0x0000FFFF, 0xFFFF0000, 0x55555555, 0x00000001) and the values equal the full-wave model, 0 of 256 wrong. Same
  rule as gfx1201.

## Results

Listing columns are READ from the `.s` files and the scans (16 of 16 scan expectations pass in each of the four builds).
Card columns are MEASURED: wrong results per launch of 256 (case 1) or of the stored lanes (case 2); 8 launches per job,
3 jobs per build; identical in all 24 launches unless said.

### Case 1: matrix result used as the next A

| Shape | Listing between the two WMMAs (all four builds) | Card: wrong per launch (all four builds) |
|---|---|---|
| Straight line, one function | `v_nop` + `s_delay_alu` (the compiler's fix) | 0 |
| Non-inlined callee, caller loads operands after the call (`h_call`, `h_call2`) | callee copies the result with `v_dual_mov` (`h_call`) or returns it as is (`h_call2`); in the caller 4 loads and an `s_waitcnt` follow the return before the WMMA | 0 |
| Non-inlined callee ending in the WMMA, caller's operands already loaded (`h_call3`: an empty asm orders them; `h_call4`: plain C++) | **nothing**: callee `v_wmma ...; s_setpc_b64`, caller `s_swappc_b64; v_wmma` | **44 of 256** in every launch, except `h_call3` in one ROCm 10.1 -O3 job (0, 0, then 44 in the other six launches); equal to the break bit for bit on one extra launch per job (0 of 256 differ) |
| Inline assembly with `v_nop` | (written) | 0 |
| **Break:** inline assembly without `v_nop` | (written; scan: 1 HAZARD) | **44 of 256 (FAIL, as the control must)** |
| Control for the probe: A replaced by the stale register's old contents | (written) | 256 of 256, equal to the stale-A prediction |

By build:

| Compiler | Opt | V_NOP in the straight chain | Missing across the call (h_call3, h_call4) | Card, call miss | Card, break |
|---|---|---|---|---|---|
| AMD clang 24 (ROCm 10.1.0, d6f6cb6) | -O3 | yes | yes | 44 (`h_call3`: 0 in the first two launches of one job) | 44 |
| AMD clang 24 (ROCm 10.1.0) | -O1 | yes | yes | 44 | 44 |
| AMD clang 24 (ROCm 10.2.0, bdc3584) | -O3 | yes | yes | 44 | 44 |
| clang 20.0.0.rocm (system ROCm 7.1.1) | -O3 | yes | yes | 44 | 44 |

The hazard-scan heuristic also flags `h_call` and `h_call2` ("CALL"), though they are right on the card: it reports
any WMMA reached from a call before a VALU, and loads and a wait are not VALU. So the card test decides, the scan only
finds sites to look at (same caveat as in the RDNA 4 study).

### Case 2: matrix builtin under a per-lane condition (valid lanes = lane < n)

| Shape | WMMA masked in the listing (scan sites) | Card: wrong results of the stored lanes |
|---|---|---|
| `sink`: operands loaded by all lanes, result stored under `if` | 0 with ROCm 10.1 (O3, O1) and 10.2; **1 with the system clang 20** | 0 of 64, 128, 160, 248 (n = 8, 16, 20, 31), every build, every launch |
| `ifload`: everything under `if (lane < n)` | 1, all builds | n = 8: 32 of 64; **n = 16: 0 of 128**; n = 20: 24 of 160; n = 31: 15 of 248 |
| `early`: early `return` for lanes >= n | 1, all builds | the same counts; n = 8 gave 32 in most launches but 26 in some (16 of 96 launches over the four builds): unstable |
| per-lane loop (lanes 0-15 two trips, 16-31 one) | 1, all builds | 128 of 256 |
| safe: padding by select (rows >= 5 and >= 16) | 0 | 0 |
| safe: wave-uniform early return | 0 | 0 |
| safe: loop with the same trip count in every lane | (same loop, 1 site) | 0 |
| Control: asm EXEC probe | (inline assembly) | all switched-off lanes written; equals the full-wave model (one launch per job) |

Control logic: unsafe shapes `ifload`, `early` and the per-lane loop FAIL, the safe forms PASS, in every launch of every
build. So the probe can see the bug.

## What it means next to the RDNA 4 result

- **Call boundary.** The RDNA 4 study's finding is not RDNA 4 specific. gfx1100 has the same hardware hazard (a WMMA that
  reads the previous WMMA's result as A gives wrong results with nothing between them and right results with one
  `v_nop` between; how few instructions suffice was not tested; 44 of 256 wrong here against 64 of 256 there) and the same compiler gap (no wait inserted across a call). MEASURED for both. Four builds, two
  compiler lines, -O1 and -O3 all show it. Counts differ because the lane layout and how much of A is reused differ.
- **Per-lane mask.** Same hardware rule (the instruction ignores EXEC, reads all lanes, writes all lanes) and the same
  wrong results for the same source shapes. gfx11 differs in which n fail: 16 valid lanes of 32 is fine here (each half
  computes its own output rows from its own lanes' copies, and with n = 16 only lanes 0-15 store) and the odd-row C in
  lanes 16-31 is not stored. The RDNA 4 study's n = 16 case was 128 of 128 wrong in the owner's run and 120 of 128 in its rerun. INFERRED from the layout; the measured n = 8, 20, 31
  failures and the n = 16 pass agree with it.
- **Sinking.** The README's gfx11 note (the compiler moved a WMMA chain under a per-lane mask) shows up with the system
  clang 20 for `sink` (the scan sees the WMMA inside the mask). It does not give wrong results here: the operands were
  loaded before the branch and the hardware ignores EXEC. ROCm 10.1 and 10.2 do not sink in this shape. The README's
  kernel had a different shape (the operands themselves were produced under the mask), so the two statements are
  consistent, not contradictory. INFERRED; I did not rebuild that kernel.
- **The compiler knows the rule on gfx1100.** The straight chain gets `v_nop` + `s_delay_alu` in all four builds
  (READ), so LLVM's WMMA hazard handling covers gfx11, which is the only direct evidence of the hazard rule for gfx11 I
  have; I could not read LLVM's `fixWMMAHazards` or AMD's RDNA 3 ISA guide on this machine (no LLVM source or guide
  installed). The measured wrong result confirms the hazard exists in the hardware here.

## What stays open

- The V_NOP count the hardware needs on gfx1100 (one VALU, or less, or more): I tested only with and without one
  `v_nop`. The `v_nop` form was right in every launch.
- A non-inlined callee that ends in a WMMA in a real kernel: none was looked for in llama.cpp or in any shipping
  gfx1100 code here (the RDNA 4 study found none in llama.cpp's gfx12 kernels).
- A different wave count per workgroup, other WMMA types (BF16, iu8) and other chain shapes (B operand, loops) were not
  tested. Only the FP16-accumulator to FP32 pair was.
- ROCm 10.2 and the system build were not run at -O1.
- Inputs were integers 0 to 3 only (non-negative), so every product is exact. Mixed-sign inputs were not used for the
  hazard counts (see Setup); whether the counts change with them is not tested.
- The bit-for-bit equality with the no-V_NOP assembly was checked on one extra launch per job, not on all launches, and
  `asm_exec_rule` (case 2) ran one launch per job.
- AMD's own statement of the gfx11 rule (ISA guide) not read.

## Rerun

```
RUN=build/hazards-runs/$(date -u +%Y%m%dT%H%M%SZ)-hazards; mkdir -p $RUN
for cfg in "rocm-10.1.0 O3" "rocm-10.2.0 O3" "system O3" "rocm-10.1.0 O1"; do set -- $cfg; scripts/hazards/build.sh $1 $2 $RUN; done   # CPU only
scripts/hazards/scan.sh $RUN                      # CPU only: both listing scans with expectations
scripts/hazards/run.sh $RUN 8 3                    # needs the 7900 XT; takes the shared GPU lock; 8 launches x 3 jobs
```

Raw output of the run described here is in `build/hazards-runs/20261126T025112Z-hazards/` (not committed; `build/` is ignored): `expect.md` (the predictions
written before the first GPU job, with two addenda), `results/`, `scans/`, `card.log`, the four builds with their
listings, and two kept-but-invalid earlier runs.
