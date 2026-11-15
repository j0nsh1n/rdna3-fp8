"""Median, min, max and spread per (type, n) from test-backend-ops perf lines in one or more logs.

Usage: python3 proto/parse_tbo.py <log> [log ...]
"""
import collections
import re
import statistics
import sys

LINE = re.compile(r"^\s+MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+).*?\s([0-9.]+) us/run")
times = collections.defaultdict(list)
for path in sys.argv[1:]:
    for ln in open(path):
        m = LINE.match(re.sub(r"\x1b\[[0-9;]*m", "", ln))
        if m:
            times[(m.group(1), int(m.group(3)))].append(float(m.group(5)))
for (t, n), v in sorted(times.items()):
    med = statistics.median(v)
    print(f"{t:8s} n={n:<4d} runs={len(v):2d} median {med:9.1f} us  min {min(v):9.1f}  max {max(v):9.1f}  "
          f"spread {(max(v) - min(v)) / med * 100:5.1f} %")
