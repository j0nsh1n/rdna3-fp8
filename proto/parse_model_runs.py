"""Summarise bench/run-model.sh logs: llama-bench (pp512, tg128) and llama-batched-bench (decode t/s per n_parallel).

Usage: python3 proto/parse_model_runs.py <run dir>
Groups by (build, model file) from the "$ command" line before each result; prints medians across rounds.
"""
import collections
import re
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])


def key(cmd):
    build = "patched" if "/build/llama.cpp/" in cmd else "base" if "/llama.cpp-base/" in cmd else "?"
    m = re.search(r"-m (\S+)", cmd)
    return build, Path(m.group(1)).name.replace("llama32-3b-", "").replace(".gguf", "")


bench = collections.defaultdict(list)
cmd = None
for ln in (run / "bench.log").read_text().splitlines() if (run / "bench.log").exists() else []:
    if ln.startswith("$ "):
        cmd = ln
    m = re.match(r"\|\s*llama.*\|\s*(pp512|tg128)\s*\|\s*([0-9.]+) ± ([0-9.]+)", ln)
    if m and cmd:
        bench[key(cmd) + (m.group(1),)].append(float(m.group(2)))
if bench:
    print("llama-bench, tokens/s, median over rounds (min-max):")
    for k in sorted({k[:2] for k in bench}):
        cells = []
        for test in ("pp512", "tg128"):
            v = bench.get(k + (test,), [])
            cells.append(f"{test} {statistics.median(v):8.1f} ({min(v):.1f}-{max(v):.1f})" if v else f"{test} -")
        print(f"  {k[0]:8s} {k[1]:10s} " + "   ".join(cells))

batched = collections.defaultdict(list)
cmd = None
for ln in (run / "batched.log").read_text().splitlines() if (run / "batched.log").exists() else []:
    if ln.startswith("$ "):
        cmd = ln
    cols = [c.strip() for c in ln.strip().strip("|").split("|")]
    if cmd and len(cols) >= 8 and cols[0].isdigit() and cols[2].isdigit():
        # PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s
        batched[key(cmd) + (int(cols[2]),)].append(float(cols[7]))
if batched:
    print("llama-batched-bench, decode tokens/s (all sequences), median over rounds:")
    ks = sorted({k[:2] for k in batched})
    ns = sorted({k[2] for k in batched})
    print("  " + " " * 20 + "".join(f"{'B=' + str(n):>10s}" for n in ns))
    for k in ks:
        print(f"  {k[0]:8s} {k[1]:10s} " + "".join(f"{statistics.median(batched[k + (n,)]):10.1f}" if batched.get(k + (n,)) else f"{'-':>10s}" for n in ns))
