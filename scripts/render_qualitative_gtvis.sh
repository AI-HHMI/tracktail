#!/usr/bin/env bash
# Re-render the trials of the paper's qualitative comparison (tracktail-paper/format_arrow.ipynb) for every method,
# hiding points only where the ground truth is occluded (render_video.py --gt-visibility), so all columns show the
# same points at every frame. The predictions are the ones of the existing renders (their render_info.json).
#
# Renders go to $RENDER_ROOT/gtvis-renders/<method>/<dataset>/<trial>/ (default RENDER_ROOT
# /groups/karashchuk/home/karashchukl/renders).
#
# Usage: bash scripts/render_qualitative_gtvis.sh

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO:$PYTHONPATH"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi
RENDER_ROOT=${RENDER_ROOT:-/groups/karashchuk/home/karashchukl/renders}
OUT=$RENDER_ROOT/gtvis-renders

RES=/groups/karashchuk/home/ruppk2/results/paper-final-v2
ARROW=/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/paper-canonicalcams-cached-100

render() {   # method dataset trial npz [extra args]
    local method=$1 dataset=$2 trial=$3 npz=$4; shift 4
    echo "  rendering $method $dataset $trial"
    $PYTHON "$REPO/scripts/render_video.py" --input-npz "$npz" --output-dir "$OUT/$method/$dataset/$trial" \
        --trails --trail-length 30 --max-points 100 "$@"
}

for spec in "dex_ycb 20200918-subject-06__20200918_114117" "cmupanoptic_3dgs tennis"; do
    read -r dataset trial <<< "$spec"
    tt=$RES/posetail-outputs/main_finetune_vzkjqrys_canonical_cached_100/$dataset/${trial}_trial.npz
    render gt        $dataset $trial "$tt" --use-gt
    render tracktail $dataset $trial "$tt" --gt-visibility
    render cotracker $dataset $trial "$RES/cotracker-outputs/paper-canonicalcams-cached-100/$dataset/test/$trial/trial/cotracker_3d.npz" --gt-visibility
    render mvtracker $dataset $trial "$RES/mvtracker-outputs/paper-canonical-cached-100-duster/$dataset/${trial}_trial.npz" --gt-visibility
    render arrow     $dataset $trial "$ARROW/$dataset/${trial}_trial.npz" --gt-visibility
done

echo "Done. Renders saved to $OUT"
