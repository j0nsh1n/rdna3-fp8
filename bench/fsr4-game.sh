#!/usr/bin/env bash
# Switch Cyberpunk 2077 between WrongGPU's setup and our FSR 4 (FP8 path) test, and back, without leaving traces.
# Usage: bench/fsr4-game.sh snapshot | use base|exact | restore | status
#   snapshot   once, game closed: sha256 of every file in bin/x64 (recursive) and the game folder's top level; copies
#              OptiScaler.ini, d3d12.dll, d3d12core.dll and the vkd3d-proton shader cache aside.
#   use base   our vkd3d-proton 5b1089c8 with stock dxil-spirv;  use exact  the same with bench/dxil-spirv-rdna3-exact-fp8.patch.
#              Sets only Dx12Upscaler = ffx and Fsr4GpuSupport = 1 (FP8) in OptiScaler.ini and clears the shader cache,
#              so shaders compiled under another DXIL_SPIRV_CONFIG are not reused. The FP8 mode itself is chosen by the
#              Steam launch option DXIL_SPIRV_CONFIG=wmma_rdna3_workaround | wmma_rdna3_exact_fp8 | wmma_rdna3_exact_fp8_rte.
#   restore    copies the saved files back, removes files that were not in the snapshot, and checks every hash.
# Rules agreed with the WrongGPU project: never touch wronggpu-setup/, the WrongGPU runtime, its ini files or plugins.
set -euo pipefail

GAME="/var/mnt/games/SteamLibrary/steamapps/common/Cyberpunk 2077"
BIN="$GAME/bin/x64"
HERE="$(cd "$(dirname "$0")/.." && pwd)/build/fsr4"
SAVE="$HERE/game-backup"
DLLS="$HERE/dlls"

game_running() { ps -eo args | grep -q "[C]yberpunk2077\.exe"; }
refuse_if_running() { if game_running; then echo "close the game first"; exit 1; fi; }

hash_list() {  # sorted "sha256  relative path" for bin/x64 (recursive) and the game folder's top-level files
    (cd "$BIN" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed 's|  \./|  bin/x64/|')
    (cd "$GAME" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum | sed 's|  \./|  |')
}

set_ini() {  # set_ini <key> <value>: replace the first "key = ..." line in OptiScaler.ini
    sed -i "0,/^$1 *=.*/s//$1 = $2/" "$BIN/OptiScaler.ini"
    grep -q "^$1 = $2\$" "$BIN/OptiScaler.ini" || { echo "could not set $1"; exit 1; }
}

case "${1:-}" in
snapshot)
    refuse_if_running
    if [[ -f "$SAVE/hashes-before.txt" ]]; then echo "snapshot exists ($SAVE); restore first or remove it by hand"; exit 1; fi
    mkdir -p "$SAVE"
    hash_list > "$SAVE/hashes-before.txt"
    cp -p "$BIN/OptiScaler.ini" "$BIN/d3d12.dll" "$BIN/d3d12core.dll" "$SAVE/"
    for c in "$GAME"/vkd3d-proton.cache*; do [[ -f $c ]] && cp -p "$c" "$SAVE/"; done
    # OptiScaler and WrongGPU rewrite their logs on every launch
    mkdir -p "$SAVE/logs"
    for l in "$BIN"/*.log "$BIN"/*.log.*; do [[ -f $l ]] && cp -p "$l" "$SAVE/logs/"; done
    echo "snapshot: $(wc -l < "$SAVE/hashes-before.txt") files hashed; saved $(ls "$SAVE" | grep -v hashes | tr '\n' ' ')"
    ;;
use)
    refuse_if_running
    [[ -f "$SAVE/hashes-before.txt" ]] || { echo "run snapshot first"; exit 1; }
    case "${2:-}" in base|exact) ;; *) echo "usage: use base|exact"; exit 2;; esac
    cp "$DLLS/$2/d3d12.dll" "$DLLS/$2/d3d12core.dll" "$BIN/"
    cp "$SAVE/OptiScaler.ini" "$BIN/OptiScaler.ini"
    set_ini Dx12Upscaler ffx
    set_ini Fsr4GpuSupport 1
    rm -f "$GAME"/vkd3d-proton.cache*
    echo "using $2 build ($(sha256sum "$BIN/d3d12core.dll" | cut -c1-12)), FSR 4 FP8 selected, vkd3d-proton cache cleared"
    ;;
restore)
    refuse_if_running
    [[ -f "$SAVE/hashes-before.txt" ]] || { echo "no snapshot"; exit 1; }
    cp -p "$SAVE/OptiScaler.ini" "$SAVE/d3d12.dll" "$SAVE/d3d12core.dll" "$BIN/"
    rm -f "$GAME"/vkd3d-proton.cache*
    for c in "$SAVE"/vkd3d-proton.cache*; do [[ -f $c ]] && cp -p "$c" "$GAME/"; done
    for l in "$SAVE"/logs/*; do [[ -f $l ]] && cp -p "$l" "$BIN/"; done
    # remove files that appeared since the snapshot (logs, screenshots stay outside bin/x64)
    hash_list | awk '{ $1=""; sub(/^ +/, ""); print }' | while IFS= read -r f; do
        grep -qF "  $f" "$SAVE/hashes-before.txt" || { echo "removing new file: $f"; rm -f "$GAME/$f"; }
    done
    if diff <(hash_list) "$SAVE/hashes-before.txt" > "$HERE/restore-diff.txt"; then
        echo "restored: every file matches the snapshot"
    else
        echo "MISMATCH after restore, see $HERE/restore-diff.txt"; exit 1
    fi
    ;;
status)
    game_running && echo "game: running" || echo "game: closed"
    echo "d3d12core.dll: $(sha256sum "$BIN/d3d12core.dll" | cut -c1-12)"
    grep -E '^(Dx12Upscaler|Fsr4GpuSupport) *=' "$BIN/OptiScaler.ini"
    ;;
*)
    echo "usage: $0 snapshot | use base|exact | restore | status"; exit 2
    ;;
esac
