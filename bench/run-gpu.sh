#!/usr/bin/env bash
# GPU runs for the FP8 prototypes on the RX 7900 XT, following notes/test-plan.md.
# Usage: bench/run-gpu.sh [phase ...]   phases: build correct baseline decode prefill (default: all, in that order)
# Logs go to build/runs/<timestamp>/, one file per phase plus summary.txt.
#
# Every timed command takes the shared GPU lock on its own, so the WrongGPU project's runs get turns between ours.
# Untimed correctness runs skip the lock (short, and overlap only disturbs timings).
# Memory: stop if available memory falls below 3 GB; timed runs wait for 8 GB (machine rules).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/home/jonathans/Projects/wronggpu/build/rdna3-analysis/gpu-bench.lock
GEMV="$ROOT/build/gemv/fp8_gemv"
GEMM="$ROOT/build/gemm/fp8_gemm"
TBO="$ROOT/build/llama.cpp/build-hip/bin/test-backend-ops"
OUT="$ROOT/build/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"

avail_gb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }

need_mem() {  # need_mem <min GB to start> ; waits up to 30 min, aborts below the 3 GB stop line
    local want=$1 waited=0
    while (( $(avail_gb) < want )); do
        if (( $(avail_gb) < 3 )); then
            echo "STOP: available memory $(avail_gb) GB is below 3 GB" | tee -a "$SUMMARY"
            exit 4
        fi
        if (( waited >= 1800 )); then
            echo "STOP: waited 30 min for $want GB available (have $(avail_gb) GB)" | tee -a "$SUMMARY"
            exit 4
        fi
        sleep 60
        waited=$((waited + 60))
    done
}

capped() {  # capped <MemoryMax> <cmd...>
    local mem=$1
    shift
    timeout 600 systemd-run --user --scope -q -p MemoryMax="$mem" -p MemorySwapMax=0 -- nice "$@"
}

untimed() {  # untimed <log> <cmd...> ; records the exit code instead of stopping the phase
    local log=$1
    shift
    need_mem 4
    echo "\$ $*" >> "$log"
    local rc=0
    capped 2G "$@" >> "$log" 2>&1 || rc=$?
    echo "exit $rc" >> "$log"
    if (( rc != 0 )); then
        echo "FAIL ($rc): $*" | tee -a "$SUMMARY"
    fi
}

timed() {  # timed <log> <cmd...>
    local log=$1
    shift
    need_mem 8
    echo "\$ $*" >> "$log"
    local rc=0
    flock "$LOCK" bash -c 'echo "clock: $(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep "\*" | head -1)"' >> "$log" 2>&1 || true
    flock "$LOCK" timeout 900 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- nice "$@" >> "$log" 2>&1 || rc=$?
    echo "exit $rc" >> "$log"
    if (( rc != 0 )); then
        echo "FAIL ($rc): $*" | tee -a "$SUMMARY"
    fi
}

TBO_SHAPE='type_b=f32,m=16384,n=NS,k=14336,bs=\[1,1\],nr=\[1,1\]'
tbo() {  # tbo <log> <type regex> <n regex>
    timed "$1" "$TBO" perf -b ROCm0 -o MUL_MAT -p "type_a=($2),${TBO_SHAPE/NS/$3}"
}

phase_build() {
    need_mem 8
    local log="$OUT/build.log"
    (cd "$ROOT/build/gemv" && capped 6G hipcc -O3 -std=c++17 --offload-arch=gfx1100 -save-temps \
        "$ROOT/proto/gemv/fp8_gemv.hip" -o fp8_gemv -L/usr/lib64 -lamdhip64) >> "$log" 2>&1
    (cd "$ROOT/build/gemm" && capped 6G hipcc -O3 -std=c++17 --offload-arch=gfx1100 -save-temps \
        "$ROOT/proto/gemm/fp8_gemm.hip" -o fp8_gemm -L/usr/lib64 -lamdhip64) >> "$log" 2>&1
    echo "build: ok" | tee -a "$SUMMARY"
}

phase_correct() {
    local log="$OUT/correct.log"
    # Ragged M (4093 is prime, not a multiple of any rows-per-block), K = 4096.
    for v in a b; do
        for r in 1 2 4; do
            for n in 1 2 3 4 5 6 7 8; do
                untimed "$log" "$GEMV" "$v" "$r" "$n" 4093 4096 2 plain
            done
            for n in 1 3 8; do
                untimed "$log" "$GEMV" "$v" "$r" "$n" 4093 4096 2 nan
            done
        done
    done
    for r in 1 2 4; do
        for n in 1 8; do
            untimed "$log" "$GEMV" b "$r" "$n" 4093 4096 2 ovf
            untimed "$log" "$GEMV" a "$r" "$n" 4093 4096 2 ovf
        done
    done
    for n in 64 100 512; do
        untimed "$log" "$GEMM" "$n" 4096 4096 2
    done
    untimed "$log" "$GEMM" 100 4096 4096 2 nan
    echo "correct: $(grep -c '^\$' "$log") runs, $(grep -c -v '^exit 0$' <(grep '^exit' "$log")) non-zero exits" | tee -a "$SUMMARY"
}

phase_baseline() {
    local log="$OUT/baseline.log"
    for i in $(seq 1 10); do
        tbo "$log" 'q8_0' '[1-8]'
    done
    for i in 1 2 3; do
        tbo "$log" 'f16|f8_e4m3' '[1-8]'
    done
    echo "baseline: done" | tee -a "$SUMMARY"
}

phase_decode() {
    local log="$OUT/decode.log"
    for round in 1 2 3; do
        tbo "$log" 'q8_0' '[1-8]'
        for v in a b; do
            for r in 1 2 4; do
                for n in 1 2 3 4 5 6 7 8; do
                    timed "$log" "$GEMV" "$v" "$r" "$n" 16384 14336 200 time
                done
            done
        done
    done
    echo "decode: done" | tee -a "$SUMMARY"
}

phase_prefill() {
    local log="$OUT/prefill.log"
    for n in 64 128 256 512 1024 2048 4096; do
        timed "$log" "$GEMM" "$n" 16384 14336 20
    done
    tbo "$log" 'f16|q8_0|f8_e4m3' '(64|128|256|512|1024|2048|4096)'
    echo "prefill: done" | tee -a "$SUMMARY"
}

phases=("$@")
if (( ${#phases[@]} == 0 )); then
    phases=(build correct baseline decode prefill)
fi
echo "run $OUT, phases: ${phases[*]}, available $(avail_gb) GB" | tee -a "$SUMMARY"
for p in "${phases[@]}"; do
    "phase_$p"
done
echo "finished" | tee -a "$SUMMARY"
