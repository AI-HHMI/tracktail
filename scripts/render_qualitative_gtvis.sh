#!/usr/bin/env bash
# Re-render the trials of the paper's qualitative comparison (tracktail-paper/figure_qualitative.ipynb) for every method,
# hiding points only where the ground truth is occluded (render_video.py --gt-visibility), so all columns show the
# same points at every frame. The tracks are colored by their order in the figure's camera at a figure frame
# (render_video.py --color-order-view/-frame/-axis), with the same colors in every column.
#
# Renders go to $RENDER_ROOT/gtvis-renders/<method>/<dataset>/<trial>/ (default RENDER_ROOT $TRACKTAIL_RESULTS/renders).
# Inputs: the *_100 runs of every method (tracktail_100.sh, cotracker_100.sh, mvtracker_100.sh, arrow_100.sh) under the
# results trees of scripts/paper_common.sh. Tracktail: the QUERY_OCC=gt run (folder ..._cached_100_occgt), as used for
# the paper figure; set TRACKTAIL_100 to use another run.
#
# Usage: bash scripts/render_qualitative_gtvis.sh

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PYTHON=$PIXI_PYTHON
RENDER_ROOT=${RENDER_ROOT:-$TRACKTAIL_RESULTS/renders}
OUT=$RENDER_ROOT/gtvis-renders

TRACKTAIL_100=${TRACKTAIL_100:-$TRACKTAIL_OUT_BASE/main_finetune_vzkjqrys_canonical_cached_100_occgt}
ARROW=$ARROW_OUT_BASE/paper-canonicalcams-cached-100

render() {   # method dataset trial npz [extra args]
    local method=$1 dataset=$2 trial=$3 npz=$4; shift 4
    echo "  rendering $method $dataset $trial"
    $PYTHON "$REPO/scripts/render_video.py" --input-npz "$npz" --output-dir "$OUT/$method/$dataset/$trial" \
        --trails --trail-length 30 --max-points 100 --color-order-view "$view" \
        --color-order-frame "$frame" --color-order-axis "$axis" "$@"
}

# dataset trial view frame axis: colors ordered by the ground truth in the figure's camera at that frame (0-based,
# -1 = last): DexYCB left to right at t = 24, Panoptic top to bottom at the last frame (t = 150)
for spec in "dex_ycb 20200918-subject-06__20200918_114117 view_00 23 x" "cmupanoptic_3dgs tennis 14 -1 y"; do
    read -r dataset trial view frame axis <<< "$spec"
    tt=$TRACKTAIL_100/$dataset/${trial}_trial.npz
    render gt        $dataset $trial "$tt" --use-gt
    render tracktail $dataset $trial "$tt" --gt-visibility
    render cotracker $dataset $trial "$COTRACKER_OUT_BASE/paper-canonicalcams-cached-100/$dataset/test/$trial/trial/cotracker_3d.npz" --gt-visibility
    render mvtracker $dataset $trial "$MVTRACKER_OUT_BASE/paper-canonical-cached-100-duster/$dataset/${trial}_trial.npz" --gt-visibility
    render arrow     $dataset $trial "$ARROW/$dataset/${trial}_trial.npz" --gt-visibility
done

echo "Done. Renders saved to $OUT"
