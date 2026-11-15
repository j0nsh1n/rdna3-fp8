"""Count instructions in each kernel's innermost backward-branch loop in an AMDGPU .s file.

Usage: python3 loop_isa.py <file.s> [kernel-name-regex]
Prints, per kernel: loop length, VALU / SALU / memory / LDS counts, per-weight VALU (from the R template argument,
16 weights per row per lane per iteration), and the opcode histogram.
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


def innermost_loop(body):
    labels = {}
    for i, ln in enumerate(body):
        m = re.match(r"^(\.LBB\S+):", ln)
        if m:
            labels[m.group(1)] = i
    best = None
    for i, ln in enumerate(body):
        m = re.match(r"^\s*s_cbranch_\w+\s+(\.LBB\S+)", ln) or re.match(r"^\s*s_branch\s+(\.LBB\S+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            start = labels[m.group(1)]
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
    return best


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
        loop = innermost_loop(body)
        if not loop:
            print(f"{name}: no loop")
            continue
        ops = []
        for ln in body[loop[0]:loop[1] + 1]:
            t = ln.strip()
            if not t or t.startswith((";", ".")) or t.endswith(":"):
                continue
            ops.append(t.split()[0])
        cls = collections.Counter(classify(o) for o in ops)
        m = re.search(r"<(\d+), (\d+)>", name)
        n, r = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        weights = 16 * r if r else 0
        hist = collections.Counter(o for o in ops if classify(o) == "valu")
        per_w = f"{cls['valu'] / weights:.2f} VALU/weight" if weights else ""
        print(f"{name}\n  loop {len(ops)} instr: valu {cls['valu']} salu {cls['salu']} vmem {cls['vmem']} "
              f"lds {cls['lds']} wait {cls['wait']}  {per_w}")
        print("  " + ", ".join(f"{k} {v}" for k, v in hist.most_common()))


if __name__ == "__main__":
    main()
