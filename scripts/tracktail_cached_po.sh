#!/usr/bin/env bash
#
# Tracktail inference + metrics on the curated 512 tracks of the Point Odyssey test sets (Tab. po_results; pose3d.npz,
# written by scripts/make_pose3d_512.py and shipped with the released data). Runs locally, same style as tracktail_cached.sh.
#   scripts/eval_testset_metrics.py   inference (query_first: each track at the first frame visible in the cameras that
#                                     are used) and the metrics (the paper protocol by default; add --legacy-metrics
#                                     there for the previous pooled definition)
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Both are run by
# default, each into its own folder:
#   $TRACKTAIL_OUT_BASE/main_finetune_vzkjqrys_po_4cams   and   $TRACKTAIL_OUT_BASE/main_finetune_vzkjqrys_po_9cams
#
# Usage:
#   bash tracktail_cached_po.sh [CAMS] [DATASETS]
#
# Examples:
#   bash tracktail_cached_po.sh
#   bash tracktail_cached_po.sh 4
#   bash tracktail_cached_po.sh "4 9" point-odyssey-human
#
# Paper protocol: QUERY_OCC=unknown (default; no GT visibility at the query frame), VIS_THRESH=0.1 (default).
# Env overrides: TRACKTAIL_OUT_BASE, NETWORK (default main_finetune_vzkjqrys), POSE_FILE (default pose3d.npz),
#   DEVICE (default cuda:0), QUERY_OCC, VIS_THRESH.

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
POSE_FILE=${POSE_FILE:-pose3d.npz}
QUERY_OCC=${QUERY_OCC:-unknown}
VIS_THRESH=${VIS_THRESH:-0.1}
NETWORK=${NETWORK:-main_finetune_vzkjqrys}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
PYTHON=$PIXI_PYTHON
MODEL_ARGS=$(model_args "$NETWORK")

cd "$REPO"

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi
    OUT="$TRACKTAIL_OUT_BASE/${NETWORK}_po_${N}cams"
    echo "starting tracktail eval on the 512 Point Odyssey tracks ($N cameras, $POSE_FILE): $DATASETS -> $OUT"

    # both datasets use the same cameras, so one call writes one summary.json / metrics.md over both
    $PYTHON scripts/eval_testset_metrics.py \
        $MODEL_ARGS \
        --root          "$TRACKTAIL_DATA" \
        --out           "$OUT" \
        --datasets      $DATASETS \
        --view-seed     0 \
        --device        "${DEVICE:-cuda:0}" \
        --pose-file     "$POSE_FILE" \
        --query-occlusion "$QUERY_OCC" \
        --vis-thresh    "$VIS_THRESH" \
        $CAM_ARG
done

echo "done!"
