#!/usr/bin/env bash
# Render ARROW predictions using render_video.py, like cotracker_render.sh and mvtracker_render.sh.
#
# ARROW structure:   <output-dir>/<dataset>/<session>_<trial>.npz   (arrow_inference.py)
#
# Renders go to $RENDER_ROOT/arrow-renders/<basename of output-dir>/<dataset>/<session>_<trial>/ (default RENDER_ROOT
# /groups/karashchuk/home/karashchukl/renders). Frames before a track's query frame are hidden (render_video.py).
# Outputs written before arrow_inference.py saved video_paths get them added from metadata_path and cam_names_used
# (in a temporary copy; the prediction file is not changed).
#
# Usage:
#   bash arrow_render.sh <output-dir> [dataset ...]
#
# Examples:
#   bash arrow_render.sh /groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/paper-canonical-cached-100
#   bash arrow_render.sh /groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/paper-canonical-cached-100 dex_ycb

set -e

OUTPUT_DIR="${1:?Usage: $0 <output-dir> [dataset ...]}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
shift
DATASETS=("${@:-*}")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO:$PYTHONPATH"

if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi
SCRIPT="$REPO/scripts/render_video.py"
RENDER_ROOT=${RENDER_ROOT:-/groups/karashchuk/home/karashchukl/renders}
RENDER_BASE=$RENDER_ROOT/arrow-renders/$(basename "$OUTPUT_DIR")

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

shopt -s nullglob
for dataset in "${DATASETS[@]}"; do
for npz in "$OUTPUT_DIR"/$dataset/*.npz; do
    [[ "$(basename "$npz")" == *_metrics.npz ]] && continue
    rel="${npz#$OUTPUT_DIR/}"                 # e.g. dex_ycb/20201015-subject-09__20201015_144721_trial.npz
    out="$RENDER_BASE/${rel%.npz}"
    input="$npz"
    # add video_paths (the img/<camera> folders of the cameras used) when the file has none
    input=$($PYTHON - "$npz" "$TMP" <<'EOF'
import os, sys
import numpy as np
npz, tmp = sys.argv[1:]
d = np.load(npz, allow_pickle=True)
if 'video_paths' in d.files:
    print(npz)
else:
    trial = os.path.dirname(str(d['metadata_path']))
    out = dict(d)
    out['video_paths'] = np.array([os.path.join(trial, 'img', str(c)) for c in d['cam_names_used']], dtype=str)
    path = os.path.join(tmp, os.path.basename(npz))
    np.savez(path, **out)
    print(path)
EOF
)
    echo "  rendering $rel"
    $PYTHON "$SCRIPT" \
        --input-npz  "$input" \
        --output-dir "$out" \
        --trails --trail-length 30 \
        --max-points 100
done
done

echo "Done. Renders saved to $RENDER_BASE"
