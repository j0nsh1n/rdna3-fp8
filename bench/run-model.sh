#!/usr/bin/env bash
# End-to-end model test: Llama 3.2 3B Instruct FP8 (RedHatAI) as FP8, FP8 with a Q8_0 output layer, Q8_0 and F16,
# on the patched llama.cpp (build/llama.cpp, bench/llama-cpp-fp8-mmvf-hip.patch) and unpatched (build/llama.cpp-base).
# Usage: bench/run-model.sh [phase ...]   phases: tools ppl kld bench batched (default: all, in that order)
# Logs: build/model-runs/<timestamp>/. Every GPU command holds the shared GPU lock and sees only device 0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/home/jonathans/Projects/wronggpu/build/rdna3-analysis/gpu-bench.lock
PATCHED="$ROOT/build/llama.cpp/build-hip/bin"
BASE="$ROOT/build/llama.cpp-base/build-hip/bin"
# MODEL selects the GGUF set: models/<MODEL>-{auto,fp8-outq8,q8_0,f16}.gguf (default llama32-3b). Missing files are skipped.
M="$ROOT/models"
MODEL="${MODEL:-llama32-3b}"
FP8="$M/$MODEL-auto.gguf"
FP8Q="$M/$MODEL-fp8-outq8.gguf"
Q8="$M/$MODEL-q8_0.gguf"
F16="$M/$MODEL-f16.gguf"
TEXT="$ROOT/build/ppl-text.txt"
OUT="$ROOT/build/model-runs/$(date +%Y%m%d-%H%M%S)-${MODEL:-llama32-3b}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
export HIP_VISIBLE_DEVICES=0

avail_gb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }

gpu() {  # gpu <log> <cmd...> : one locked, memory-capped GPU command; skipped if its -m model file is missing
    local log=$1
    shift
    local a prev=""
    for a in "$@"; do
        if [[ $prev == -m && ! -f $a ]]; then
            echo "skip (no model file $a): $*" >> "$log"
            return 0
        fi
        prev=$a
    done
    local waited=0
    while (( $(avail_gb) < 8 )); do
        if (( $(avail_gb) < 3 || waited >= 1800 )); then
            echo "STOP: available memory $(avail_gb) GB" | tee -a "$SUMMARY"
            exit 4
        fi
        sleep 60
        waited=$((waited + 60))
    done
    echo "\$ $*" >> "$log"
    local rc=0
    flock "$LOCK" timeout 1800 systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 -- nice "$@" >> "$log" 2>&1 || rc=$?
    echo "exit $rc" >> "$log"
    if (( rc != 0 )); then
        echo "FAIL ($rc): $*" | tee -a "$SUMMARY"
    fi
}

phase_tools() {
    for d in "$ROOT/build/llama.cpp" "$ROOT/build/llama.cpp-base"; do
        systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- nice \
            cmake --build "$d/build-hip" -j 8 --target llama-bench llama-perplexity llama-batched-bench >> "$OUT/tools.log" 2>&1
    done
    if [[ ! -s "$TEXT" ]]; then
        cat "$ROOT"/build/llama.cpp-base/docs/*.md "$ROOT"/build/llama.cpp-base/docs/*/*.md > "$TEXT"
    fi
    echo "tools: ok, text $(wc -c < "$TEXT") bytes, sha256 $(sha256sum "$TEXT" | cut -c1-16)" | tee -a "$SUMMARY"
}

phase_ppl() {
    local log="$OUT/ppl.log"
    for bin in "$PATCHED" "$BASE"; do
        for m in "$FP8" "$FP8Q" "$Q8" "$F16"; do
            echo "== $(basename "$(dirname "$(dirname "$bin")")") $(basename "$m")" >> "$log"
            gpu "$log" "$bin/llama-perplexity" -m "$m" -f "$TEXT" -c 512 --chunks 50 -ngl 99
        done
    done
    echo "ppl: done" | tee -a "$SUMMARY"
}

phase_kld() {
    # Reference: unpatched build with 8-token batches (llama.cpp's own FP32 decode path; large batches are the NaN bug).
    # Small batches send every matrix multiply through the patched decode kernels (1 = batch-1 path, 3 and 8 = rows kernel).
    local log="$OUT/kld.log" logits="$ROOT/build/kld-base-fp8.bin"
    echo "== base FP8 logits, ub 8" >> "$log"
    gpu "$log" "$BASE/llama-perplexity" -m "$FP8" -f "$TEXT" -c 512 --chunks 10 -ngl 99 -b 8 -ub 8 --kl-divergence-base "$logits"
    for ub in 8 3 1; do
        echo "== patched FP8 ub $ub vs base FP8 ub 8" >> "$log"
        gpu "$log" "$PATCHED/llama-perplexity" -m "$FP8" -f "$TEXT" -c 512 --chunks 10 -ngl 99 -b "$ub" -ub "$ub" --kl-divergence-base "$logits" --kl-divergence
    done
    echo "== patched FP8 ub 512 (hipBLAS path with the F32-output fix) vs base FP8 ub 8" >> "$log"
    gpu "$log" "$PATCHED/llama-perplexity" -m "$FP8" -f "$TEXT" -c 512 --chunks 10 -ngl 99 --kl-divergence-base "$logits" --kl-divergence
    echo "== Q8_0 vs base FP8 ub 8 (for scale)" >> "$log"
    gpu "$log" "$PATCHED/llama-perplexity" -m "$Q8" -f "$TEXT" -c 512 --chunks 10 -ngl 99 --kl-divergence-base "$logits" --kl-divergence
    rm -f "$logits"
    echo "kld: done" | tee -a "$SUMMARY"
}

phase_bench() {
    local log="$OUT/bench.log"
    for round in 1 2 3; do
        for m in "$Q8" "$FP8Q" "$FP8" "$F16"; do
            gpu "$log" "$PATCHED/llama-bench" -m "$m" -ngl 99 -p 512 -n 128 -r 3 -o md
        done
        for m in "$FP8Q" "$FP8"; do
            gpu "$log" "$BASE/llama-bench" -m "$m" -ngl 99 -p 512 -n 128 -r 3 -o md
        done
    done
    echo "bench: done" | tee -a "$SUMMARY"
}

phase_batched() {
    local log="$OUT/batched.log"
    for round in 1 2; do
        for m in "$Q8" "$FP8Q"; do
            gpu "$log" "$PATCHED/llama-batched-bench" -m "$m" -ngl 99 -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,3,4,5,6,7,8
        done
        gpu "$log" "$BASE/llama-batched-bench" -m "$FP8Q" -ngl 99 -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,3,4,5,6,7,8
    done
    echo "batched: done" | tee -a "$SUMMARY"
}

phases=("$@")
if (( ${#phases[@]} == 0 )); then
    phases=(tools ppl kld bench batched)
fi
echo "run $OUT, phases: ${phases[*]}, available $(avail_gb) GB" | tee -a "$SUMMARY"
for p in "${phases[@]}"; do
    "phase_$p"
done
echo "finished" | tee -a "$SUMMARY"
