#!/usr/bin/env bash
# Render Tracktail predictions using render_video.py.
#
# Structure: <output-dir>/<dataset>/<trial>.npz
#
# Renders go to $RENDER_ROOT (default $TRACKTAIL_RESULTS/renders). Frames before a
# track's query frame are hidden (render_video.py).
#
# Usage:
#   bash tracktail_render.sh <output-dir> [dataset ...]
#
# Examples:
#   bash scripts/tracktail_render.sh $TRACKTAIL_RESULTS/tracktail-outputs/main_finetune_vzkjqrys_canonical_cached_100
#   bash scripts/tracktail_render.sh <output-dir> dex_ycb

set -e

OUTPUT_DIR="${1:?Usage: $0 <output-dir> [dataset ...]}"
shift
DATASETS=("${@:-*}")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PYTHON=$PIXI_PYTHON
SCRIPT="$REPO/scripts/render_video.py"
RENDER_ROOT=${RENDER_ROOT:-$TRACKTAIL_RESULTS/renders}
RENDER_BASE=$RENDER_ROOT/tracktail-renders/$(basename "$OUTPUT_DIR")

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
