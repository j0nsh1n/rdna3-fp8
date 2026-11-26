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
    # scanner exits 1 on an EXPECT FAIL, so the exit code alone cannot mean "crashed": a run counts as a scanner error
    # only when it exits above 1 or printed no EXPECT line at all.
    runscan() { # <filter-regex> <script> <file> <expect-file>
      local out rc
      out=$(python3 -I "$here/$2" "$3" "$4" 2>&1); rc=$?
      if [ "$rc" -gt 1 ] || ! grep -q '^EXPECT' <<< "$out"; then
        echo "SCAN-ERROR: $2 on $3 exited $rc without a usable result"; grep -v '^EXPECT' <<< "$out" | head -n 5
      fi
      grep -E "$1" <<< "$out" || true
    }
    # Keep the function name, drop its demangled parameter list and the trailing "(after ...)" note.
    runscan "^EXPECT|^(HAZARD|RETURN|CALL) " haz_hazscan.py "$b/haz3_hazard.s" "$here/expect/hazscan-haz3_hazard.txt" \
      | sed -E 's/^((HAZARD|RETURN|CALL) [^(]+)\([^:]*\) line /\1 line /; s/ \(after.*//'
    runscan "^EXPECT" haz_maskscan.py "$b/haz3_mask.s" "$here/expect/maskscan-haz3_mask.txt"
    runscan "^EXPECT" haz_maskscan.py "$b/haz3_mask.s" "$rd/scans/$n-sink.expect"
  } > "$rd/scans/$n.txt" 2>&1
  echo "$n: $(grep -c 'EXPECT PASS' "$rd/scans/$n.txt") pass, $(grep -c 'EXPECT FAIL' "$rd/scans/$n.txt") fail"
  grep -q -E 'EXPECT FAIL|SCAN-ERROR' "$rd/scans/$n.txt" && { bad=1; grep -q SCAN-ERROR "$rd/scans/$n.txt" && echo "$n: SCAN-ERROR x$(grep -c SCAN-ERROR "$rd/scans/$n.txt") (see $rd/scans/$n.txt)"; }
done
exit ${bad:-0}
