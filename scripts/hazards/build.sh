#!/usr/bin/env bash
# Usage: build.sh <toolchain: rocm-10.1.0|rocm-10.2.0|system> <opt: O1|O3> <outdir>
# Builds both probes for gfx1100 (CPU only) and writes the device assembly listing next to each binary.
set -euo pipefail
tc=$1; opt=$2; out=$3; extra=""
here=$(cd "$(dirname "$0")/../.." && pwd)
case $tc in
  system) hipcc=/usr/bin/hipcc; extra="-L/usr/lib64 -lamdhip64" ;;
  *) hipcc=$HOME/.local/opt/$tc/bin/hipcc ;;
esac
mkdir -p "$out/$tc-$opt"
for p in haz3_hazard haz3_mask; do
  nice "$hipcc" -$opt --offload-arch=gfx1100 -DHARNESS_ARCH='"gfx1100"' "$here/src/hazards/$p.hip" $extra -o "$out/$tc-$opt/$p"
  nice "$hipcc" -$opt --offload-arch=gfx1100 --cuda-device-only -S -DHARNESS_ARCH='"gfx1100"' "$here/src/hazards/$p.hip" -o "$out/$tc-$opt/$p.s"
done
echo "built $tc $opt: $("$hipcc" --version | sed -n 2p)"
