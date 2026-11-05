#!/usr/bin/env bash
# Tracktail n-camera ablation on dex_ycb. Runs directly (no bsub), same style as tracktail_cached.sh.
#
# N = 1 ... 8 cameras, nested: view_00 ... view_{N-1} (N = 4 is the canonical set of the MVTracker tables). Every N
# runs on its own track file, pose3d_ncam{N}_512.npz (scripts/make_pose3d_ncam.py), which the CoTracker and MVTracker
# ablations use too: 512 tracks visible in at least 2 frames in any of the N cameras; a quarter queried at a random
# visible frame, the rest at their first visible frame (MVTracker protocol). Paper-protocol metrics.
#
# Usage:
#   bash scripts/eval_ncam_ablation_local.sh [N ...]      default: 1 2 3 4 5 6 7 8
#   (scripts/eval_ncam_ablation.sh submits one LSF job per N instead)
#
# Results: $OUT_BASE/nviews-N/, and ncam_ablation_summary.csv over all N at the end
#   (default OUT_BASE=$TRACKTAIL_RESULTS/tracktail-outputs/ncam_ablation_dexycb).
# Paper protocol: QUERY_OCC=unknown (default), VIS_THRESH=0.1 (default).
# Env overrides: OUT_BASE, NETWORK, DEVICE (default cuda:0), QUERY_OCC, VIS_THRESH.

set -e

NS=${*:-"1 2 3 4 5 6 7 8"}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
PYTHON=$PIXI_PYTHON
NETWORK=${NETWORK:-main_finetune_vzkjqrys}
MODEL_ARGS=$(model_args "$NETWORK")
QUERY_OCC=${QUERY_OCC:-unknown}
VIS_THRESH=${VIS_THRESH:-0.1}
OUT_BASE=${OUT_BASE:-$TRACKTAIL_OUT_BASE/ncam_ablation_dexycb}

cd "$REPO"

for N in $NS; do
    CAMS=$(for i in $(seq 0 $((N-1))); do printf "view_%02d " $i; done)
    echo ""
    echo "=== n_views=${N} (${CAMS}) ==="
    $PYTHON scripts/eval_testset_metrics.py \
        $MODEL_ARGS \
        --root          "$TRACKTAIL_DATA" \
        --out           "$OUT_BASE/nviews-${N}" \
        --datasets      dex_ycb \
        --view-seed     0 \
        --device        "${DEVICE:-cuda:0}" \
        --pose-file     "pose3d_ncam${N}_512.npz" \
        --query-occlusion "$QUERY_OCC" \
        --vis-thresh    "$VIS_THRESH" \
        --cam-names     $CAMS
done

echo ""
echo "=== Combining all n_views ==="
$PYTHON cotracker_baseline/combine_ncam_ablation.py "$OUT_BASE"
