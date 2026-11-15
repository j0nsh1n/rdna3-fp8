"""Compare our decode prototype timings with llama.cpp's Q8_0 from one run folder.

Usage: python3 proto/parse_runs.py <run dir>
Reads baseline.log and decode.log. For each (variant, R, n): median of our per-run medians across rounds, the
Q8_0 median over every Q8_0 run in both logs, the ratio, and whether the gap exceeds Q8_0's measured spread.
"""
import collections
import re
import statistics
import sys
from pathlib import Path

ESC = re.compile(r"\x1b\[[0-9;]*m")
TBO = re.compile(r"^\s+MUL_MAT\(type_a=(\w+),type_b=f32,m=16384,n=(\d+),k=14336.*?\s([0-9.]+) us/run")
OURS_HEAD = re.compile(r"^variant (\w) R=(\d) N=(\d) M=16384 K=14336 mode=time")
OURS_TIME = re.compile(r"^median ([0-9.]+) us")

run = Path(sys.argv[1])
base = collections.defaultdict(list)
ours = collections.defaultdict(list)
for name in ("baseline.log", "decode.log"):
    path = run / name
    if not path.exists():
        continue
    key = None
    for ln in path.read_text().splitlines():
        ln = ESC.sub("", ln)
        if m := TBO.match(ln):
            base[(m.group(1), int(m.group(2)))].append(float(m.group(3)))
        elif m := OURS_HEAD.match(ln):
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
        elif (m := OURS_TIME.match(ln)) and key:
            ours[key].append(float(m.group(1)))
            key = None

weights = 16384 * 14336
print(f"{'n':>2} {'Q8_0 us':>8} {'spread':>7} | " + " | ".join(f"{v}R{r} us  ratio" for v in "ab" for r in (1, 2, 4)))
for n in range(1, 9):
    q = base.get(("q8_0", n), [])
    if not q:
        continue
    qm = statistics.median(q)
    spread = (max(q) - min(q)) / qm
    cells = []
    for v in "ab":
        for r in (1, 2, 4):
            t = ours.get((v, r, n))
            if not t:
                cells.append(f"{'-':>8} {'':>6}")
                continue
            tm = statistics.median(t)
            cells.append(f"{tm:8.1f} {tm / qm:5.3f}")
    print(f"{n:2d} {qm:8.1f} {spread * 100:6.1f}% | " + " | ".join(cells))
print()
best = {}
for (v, r, n), t in ours.items():
    tm = statistics.median(t)
    if (v, n) not in best or tm < best[(v, n)][0]:
        best[(v, n)] = (tm, r, len(t), (max(t) - min(t)) / tm)
print("best rows-per-wave per variant (weight read GB/s; verdict vs Q8_0 x 1.03, or vs spread where Q8_0 spread > 3 %)")
for v in "ab":
    for n in range(1, 9):
        if (v, n) not in best or ("q8_0", n) not in base:
            continue
        tm, r, k, sp = best[(v, n)]
        q = base[("q8_0", n)]
        qm = statistics.median(q)
        qs = (max(q) - min(q)) / qm
        gate = max(0.03, qs)
        verdict = "PASS" if tm <= qm * (1 + gate) else "FAIL"
        print(f"  {v} n={n} R={r}: {tm:7.1f} us ({weights / tm / 1e3:5.0f} GB/s), Q8_0 {qm:6.1f} us, ratio {tm / qm:5.3f}, "
              f"gate +{gate * 100:.1f} %, rounds {k}, our spread {sp * 100:.1f} %: {verdict}")
