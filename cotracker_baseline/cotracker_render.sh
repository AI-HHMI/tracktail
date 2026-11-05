#!/usr/bin/env bash
# Render CoTracker predictions using render_video.py.
#
# Renders go to $RENDER_ROOT (default $TRACKTAIL_RESULTS/renders) so the earlier renders are kept. Frames before a
# track's query frame are hidden (render_video.py).
#
# Usage:
#   bash cotracker_render.sh <output-dir> [dataset ...]
#
# Example (all datasets):
#   bash cotracker_render.sh $COTRACKER_OUT_BASE/paper-canonicalcams-cached-100
# Example (specific datasets):
#   bash cotracker_render.sh $COTRACKER_OUT_BASE/paper-canonicalcams-cached-100 dex_ycb

set -e

OUTPUT_DIR="${1:?Usage: $0 <output-dir> [dataset ...]}"
shift
DATASETS=("${@:-*}")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=$PIXI_PYTHON
SCRIPT="$REPO/scripts/render_video.py"
RENDER_ROOT=${RENDER_ROOT:-$TRACKTAIL_RESULTS/renders}
RENDER_BASE=$RENDER_ROOT/cotracker-renders/$(basename "$OUTPUT_DIR")

for dataset in "${DATASETS[@]}"; do
for npz in "$OUTPUT_DIR"/$dataset/test/*/*/cotracker_3d.npz; do
    [[ -f "$npz" ]] || continue
    rel="${npz#$OUTPUT_DIR/}"       # e.g. kubric-multiview/test/21/trial/cotracker_3d.npz
    trial_dir="$(dirname "$rel")"   # e.g. kubric-multiview/test/21/trial
    out="$RENDER_BASE/$trial_dir"
    echo "  rendering $trial_dir"
    $PYTHON "$SCRIPT" \
        --input-npz  "$npz" \
        --output-dir "$out" \
        --trails --trail-length 30 \
        --max-points 100
done
done

echo "Done. Renders saved to $RENDER_BASE"
