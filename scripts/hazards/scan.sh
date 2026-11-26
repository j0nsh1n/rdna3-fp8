#!/usr/bin/env bash
# Usage: scan.sh <rundir>   CPU only. Runs both listing scans (with expectations) on every build in <rundir>.
set -uo pipefail
rd=$1; here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$rd/scans"
for b in "$rd"/*/; do
  b=${b%/}; n=$(basename "$b"); [ -f "$b/haz3_hazard.s" ] || continue
  sink=0; [ "${n%%-*}" = system ] && sink=1
  printf '^k_sink wmma == %s\n' "$sink" > "$rd/scans/$n-sink.expect"
  {
    python3 -I "$here/haz_hazscan.py" "$b/haz3_hazard.s" "$here/expect/hazscan-haz3_hazard.txt" | grep -E "^EXPECT|^(HAZARD|RETURN|CALL) " | sed -E 's/\((_Float16|float) __vector[^ ]* //g; s/ \(after.*//' || echo "SCAN-ERROR: a scan did not run or printed nothing"
    python3 -I "$here/haz_maskscan.py" "$b/haz3_mask.s" "$here/expect/maskscan-haz3_mask.txt" | grep -E "^EXPECT" || echo "SCAN-ERROR: a scan did not run or printed nothing"
    python3 -I "$here/haz_maskscan.py" "$b/haz3_mask.s" "$rd/scans/$n-sink.expect" | grep -E "^EXPECT" || echo "SCAN-ERROR: a scan did not run or printed nothing"
  } > "$rd/scans/$n.txt" 2>&1
  echo "$n: $(grep -c 'EXPECT PASS' "$rd/scans/$n.txt") pass, $(grep -c 'EXPECT FAIL' "$rd/scans/$n.txt") fail"
  grep -q -E 'EXPECT FAIL|SCAN-ERROR' "$rd/scans/$n.txt" && { bad=1; grep -c SCAN-ERROR "$rd/scans/$n.txt" | sed "s/^/$n: scan errors: /"; }
done
exit ${bad:-0}
