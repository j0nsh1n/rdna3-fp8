"""Count instructions in every math loop of each kernel in an AMDGPU .s file.

Usage: python3 loop_isa.py <file.s> [kernel-name-regex]
A math loop is a backward-branch loop containing v_fma_mix_f32, v_dot2*_f32_f16, v_wmma or FP32 FMAs. A kernel can hold several
(variant B carries variant A's loop as its overflow fallback), so each is printed, tagged by its multiply instruction.
Per loop: VALU / SALU / memory / LDS counts, per-weight VALU (from the R template argument, 16 weights per row per lane
per iteration), and the opcode histogram.
"""
import collections
import re
import subprocess
import sys


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out))


def kernels(lines):
    cur, body = None, []
    for ln in lines:
        m = re.match(r"^(_Z\S+):\s*(;.*)?$", ln)
        if m:
            if cur:
                yield cur, body
            cur, body = m.group(1), []
            continue
        if cur and re.match(r"^\s*\.Lfunc_end", ln):
            yield cur, body
            cur, body = None, []
            continue
        if cur:
            body.append(ln)


MATH = ("v_fma_mix_f32", "v_dot2", "v_dual_dot2", "v_wmma", "v_dual_fmac_f32", "v_fmac_f32", "v_fma_f32")


def ops_in(body, start, end):
    ops = []
    for ln in body[start:end + 1]:
        t = ln.strip()
        if not t or t.startswith((";", ".")) or t.endswith(":"):
            continue
        ops.append(t.split()[0])
    return ops


def math_loops(body):
    labels = {}
    for i, ln in enumerate(body):
        m = re.match(r"^(\.LBB\S+):", ln)
        if m:
            labels[m.group(1)] = i
    loops = []
    for i, ln in enumerate(body):
        m = re.match(r"^\s*s_cbranch_\w+\s+(\.LBB\S+)", ln) or re.match(r"^\s*s_branch\s+(\.LBB\S+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            ops = ops_in(body, labels[m.group(1)], i)
            if any(o.startswith(MATH) for o in ops):
                loops.append(ops)
    # A backward branch can span several loops (layout); keep only loops that contain no other math loop.
    keep = []
    for ops in loops:
        if not any(o is not ops and len(o) < len(ops) and set(o) <= set(ops) and _contains(ops, o) for o in loops):
            keep.append(ops)
    return keep


def _contains(big, small):
    n = len(small)
    return any(big[i:i + n] == small for i in range(len(big) - n + 1))


def classify(op):
    if op.startswith(("v_", )):
        return "valu"
    if op.startswith(("s_waitcnt", "s_nop", "s_delay_alu", "s_clause")):
        return "wait"
    if op.startswith("s_"):
        return "salu"
    if op.startswith(("global_", "buffer_", "flat_", "scratch_")):
        return "vmem"
    if op.startswith("ds_"):
        return "lds"
    return "other"


def main():
    path = sys.argv[1]
    pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    lines = open(path).read().splitlines()
    ks = list(kernels(lines))
    names = demangle([k for k, _ in ks])
    for mangled, body in ks:
        name = names[mangled]
        if pat and not pat.search(name):
            continue
        loops = math_loops(body)
        if not loops:
            print(f"{name}: no math loop")
            continue
        m = re.search(r"<(\d+), (\d+)>", name)
        r = int(m.group(2)) if m else 0
        weights = 16 * r
        print(name)
        for ops in loops:
            cls = collections.Counter(classify(o) for o in ops)
            hist = collections.Counter(o for o in ops if classify(o) == "valu")
            kind = next((k for k in MATH if any(o.startswith(k) for o in ops)), "?")
            per_w = f"{cls['valu'] / weights:.2f} VALU/weight" if weights else ""
            print(f"  [{kind}] loop {len(ops)} instr: valu {cls['valu']} salu {cls['salu']} vmem {cls['vmem']} "
                  f"lds {cls['lds']} wait {cls['wait']}  {per_w}")
            print("    " + ", ".join(f"{k} {v}" for k, v in hist.most_common()))

if __name__ == "__main__":
    main()
