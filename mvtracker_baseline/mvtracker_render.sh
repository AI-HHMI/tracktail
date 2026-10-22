#!/usr/bin/env bash
# Render MVTracker predictions using render_video.py.
# Works for any config dir under mvtracker-outputs/ (native or Point Odyssey).
#
# Native structure:   <dataset>/test/<trial>/mvtracker_3d.npz
# PO structure:       <dataset>/<trial>.npz
#
# Renders go to $RENDER_ROOT (default /home/ruppk2@hhmi.org/renders-v2) so the earlier renders are kept. Frames before a
# track's query frame are hidden (render_video.py).
#
# Usage:
#   bash mvtracker_render.sh <output-dir> [dataset ...]
#
# Examples:
#   bash mvtracker_render.sh /groups/karashchuk/home/ruppk2/results/paper/mvtracker-outputs/native-4cams-duster
#   bash mvtracker_render.sh /groups/karashchuk/home/ruppk2/results/paper/mvtracker-outputs/native-4cams-duster dex-ycb-multiview
#   bash mvtracker_render.sh /groups/karashchuk/home/ruppk2/results/paper/mvtracker-outputs/point-odyssey-human-4cams-gt

set -e

OUTPUT_DIR="${1:?Usage: $0 <output-dir> [dataset ...]}"
shift
DATASETS=("${@:-*}")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:$REPO:$PYTHONPATH"

if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON=/groups/karashchuk/home/ruppk2/miniforge3/envs/mvtracker/bin/python; fi
SCRIPT="$REPO/scripts/render_video.py"
RENDER_ROOT=${RENDER_ROOT:-/home/ruppk2@hhmi.org/renders-v2}
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
