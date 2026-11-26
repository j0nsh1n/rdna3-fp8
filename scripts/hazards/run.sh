#!/usr/bin/env bash
# Usage: run.sh <rundir> [launches=8] [reps=3]
# Runs every built probe on the card, one job at a time under the shared GPU lock, each under 50 s.
# Refuses unless the 7900 XT (1002:744c) is installed. Writes <rundir>/card.log and <rundir>/results/<build>-<prog>-<rep>.txt.
# Reads sysfs only; no writes, no sudo.
set -uo pipefail
rd=$1; launches=${2:-8}; reps=${3:-3}
# HAZ_GPU_LOCK: a lock file shared with every other GPU job on the machine, so timing jobs never overlap.
LOCK=${HAZ_GPU_LOCK:-$PWD/build/gpu.lock}
mkdir -p "$(dirname "$LOCK")"
lspci -nn | grep '1002:744c' > /dev/null || { echo "7900 XT (744c) not installed; stopping" >&2; exit 2; }
sclk() { for c in /sys/class/drm/card[0-9]; do [ "$(cat $c/device/device 2>/dev/null)" = 0x744c ] && tr '\n' ' ' < $c/device/pp_dpm_sclk; done; }
mkdir -p "$rd/results"
{
  echo "date $(date -u +%FT%TZ)"; echo "kernel $(uname -r)"; echo "rpm $(rpm -q rocm-hip)"
  echo "pp_dpm_sclk before: $(sclk)"
} >> "$rd/card.log"
for b in "$rd"/*/; do
  b=${b%/}; name=$(basename "$b"); [ -x "$b/haz3_hazard" ] || continue
  for prog in haz3_hazard haz3_mask; do
    for r in $(seq 1 "$reps"); do
      # -w 60: a lock held longer than that is reported, not waited for (the brief's limit is 10 minutes).
      flock -w 60 -E 75 "$LOCK" nice timeout 50 "$b/$prog" "$launches" > "$rd/results/$name-$prog-$r.txt" 2>&1
      rc=$?
      echo "$name $prog rep$r exit=$rc" >> "$rd/card.log"
      # 75: the shared lock stayed busy for 60 s; stop instead of reading a missing run as a probe result.
      [ $rc -eq 75 ] && { echo "lock busy for 60 s; stopped" | tee -a "$rd/card.log" >&2; exit 75; }
      [ $rc -eq 124 ] && { echo "TIMEOUT $name $prog rep$r: killed at 50 s, result incomplete" | tee -a "$rd/card.log" >&2; timeouts=1; }
    done
  done
done
echo "pp_dpm_sclk after: $(sclk)" >> "$rd/card.log"
if klog=$(journalctl -k --since "-10min" 2>&1); then
  printf '%s\n' "$klog" | grep -i -E "ring .* timeout|GPU reset" >> "$rd/card.log" || echo "no ring timeout / GPU reset in kernel log" >> "$rd/card.log"
else
  echo "WARNING: kernel log could not be read (journalctl -k failed: $(printf '%s' "$klog" | head -n 1)); ring timeout / GPU reset NOT checked" | tee -a "$rd/card.log" >&2
fi
exit ${timeouts:-0}
