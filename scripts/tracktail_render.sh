#!/usr/bin/env bash
# Render Tracktail predictions using render_video.py.
#
# Structure: <output-dir>/<dataset>/<trial>.npz
#
# Renders go to $RENDER_ROOT (default /home/ruppk2@hhmi.org/renders-v2) so the earlier renders are kept. Frames before a
# track's query frame are hidden (render_video.py).
#
# Usage:
#   bash tracktail_render.sh <output-dir> [dataset ...]
#
# Examples:
#   bash tracktail_render.sh /groups/karashchuk/home/ruppk2/results/paper/posetail-outputs/main_finetune_vzkjqrys
#   bash tracktail_render.sh /groups/karashchuk/home/ruppk2/results/paper/posetail-outputs/main_finetune_vzkjqrys dex_ycb

set -e

OUTPUT_DIR="${1:?Usage: $0 <output-dir> [dataset ...]}"
shift
DATASETS=("${@:-*}")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:$REPO:$PYTHONPATH"

if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON=/groups/karashchuk/home/ruppk2/miniforge3/envs/mvtracker/bin/python; fi
SCRIPT="$REPO/scripts/render_video.py"
RENDER_ROOT=${RENDER_ROOT:-/home/ruppk2@hhmi.org/renders-v2}
RENDER_BASE=$RENDER_ROOT/posetail-renders/$(basename "$OUTPUT_DIR")

shopt -s nullglob
for dataset in "${DATASETS[@]}"; do
    for npz in "$OUTPUT_DIR"/$dataset/*.npz; do
        [[ "$(basename "$npz")" == eval_metrics.npz ]] && continue
        rel="${npz#$OUTPUT_DIR/}"
        trial_dir="$(dirname "$rel")"
        out="$RENDER_BASE/$trial_dir/$(basename "${npz%.npz}")"
        echo "  rendering $rel"
        $PYTHON "$SCRIPT" \
            --input-npz  "$npz" \
            --output-dir "$out" \
            --trails --trail-length 30 \
            --max-points 100
    done
done

echo "Done. Renders saved to $RENDER_BASE"
