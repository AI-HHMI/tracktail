#!/usr/bin/env bash
# Render MVTracker predictions using render_video.py.
# Works for any config dir under mvtracker-outputs/ (native or Point Odyssey).
#
# Native structure:   <dataset>/test/<trial>/mvtracker_3d.npz
# PO structure:       <dataset>/<trial>.npz
#
# Renders go to $RENDER_ROOT (default $TRACKTAIL_RESULTS/renders) so the earlier renders are kept. Frames before a
# track's query frame are hidden (render_video.py).
#
# Usage:
#   bash mvtracker_render.sh <output-dir> [dataset ...]
#
# Examples:
#   bash mvtracker_render.sh $MVTRACKER_OUT_BASE/paper-canonical-cached-100-duster
#   bash mvtracker_render.sh $MVTRACKER_OUT_BASE/paper-canonical-cached-100-duster dex_ycb
#   bash mvtracker_render.sh $MVTRACKER_OUT_BASE/paper-po-4cams-gt_0.5

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
RENDER_BASE=$RENDER_ROOT/mvtracker-renders/$(basename "$OUTPUT_DIR")

shopt -s globstar nullglob
for dataset in "${DATASETS[@]}"; do
for npz in "$OUTPUT_DIR"/$dataset/**/*.npz; do
    # Skip per-trial metric files — only render trajectory outputs
    [[ "$(basename "$npz")" == eval_metrics.npz ]] && continue
    [[ "$(basename "$npz")" == *_metrics.npz ]] && continue
    [[ -f "$npz" ]] || continue
    rel="${npz#$OUTPUT_DIR/}"      # e.g. kubric-multiview/test/001/mvtracker_3d.npz
                                   #   or point-odyssey-human/hum0_trial.npz
    trial_dir="$(dirname "$rel")"  # e.g. kubric-multiview/test/001
                                   #   or point-odyssey-human
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
