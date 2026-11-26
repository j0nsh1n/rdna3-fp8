#!/usr/bin/env bash
# Usage: scan.sh <rundir>   CPU only. Runs both listing scans (with expectations) on every build in <rundir>.
#        scan.sh --probe <filter-regex> <script> <file> <expect-file>
#          one scanner, so a crash stub outside this directory can be checked the same way as the loop.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)

# Exit 1 is a mismatch only when the scanner printed EXPECT FAIL.
# Any other nonzero status is a crash, including an exception after an early EXPECT PASS.
runscan() { # <filter-regex> <script> <file> <expect-file>
  local out rc script
  script=$2
  [[ $script == */* ]] || script=$here/$script
  out=$(python3 -I "$script" "$3" "$4" 2>&1); rc=$?
  if [[ $rc -ne 0 ]] && { [[ $rc -ne 1 ]] || ! grep -q '^EXPECT FAIL' <<< "$out"; }; then
    echo "SCAN-ERROR: $2 on $3 exited $rc without a usable result"
    grep -v '^EXPECT' <<< "$out" | head -n 5
  elif ! grep -q '^EXPECT' <<< "$out"; then
    echo "SCAN-ERROR: $2 on $3 exited $rc without a usable result"
  fi
  grep -E "$1" <<< "$out" || true
}

if [[ ${1:-} == --probe ]]; then
  shift
  [[ $# -eq 4 ]] || { echo "usage: scan.sh --probe <filter> <script> <file> <expect>" >&2; exit 2; }
  out=$(runscan "$@" 2>&1)
  printf '%s\n' "$out"
  grep -q SCAN-ERROR <<<"$out" && exit 1
  exit 0
fi

rd=$1
mkdir -p "$rd/scans"
for b in "$rd"/*/; do
  b=${b%/}; n=$(basename "$b"); [ -f "$b/haz3_hazard.s" ] || continue
  sink=0; [ "${n%%-*}" = system ] && sink=1
  printf '^k_sink wmma == %s\n' "$sink" > "$rd/scans/$n-sink.expect"
  {
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
