#!/usr/bin/env bash
#
# Tracktail inference + metrics on MVTracker's cached tracks (pose3d_cached.npz, written by make_pose3d_cached.py),
# on the canonical cameras of the MVTracker tables (Tab. main_results Tracktail row; ablation tables). Runs locally.
#   1. scripts/eval_testset_metrics.py        inference (query_first: each track at the frame given by the file) and the
#                                             paper-protocol metrics (per track, query frame and earlier frames not
#                                             evaluated, static/dynamic mean on kubric / dex_ycb)
#   2. scripts/merge_eval_summaries.py        one summary.json / metrics.md over all datasets
# For every network in NETWORKS (default: the paper model only). The paper model loads from Hugging Face; the pretrain
# and ablation networks are not released and load from $TRACKTAIL_MODELS/<network> (your own training runs; see
# scripts/paper_common.sh). TRACKTAIL_WANDB_BASE loads the original W&B run dirs instead.
#
# Usage:
#   bash scripts/tracktail_cached.sh [DATASETS]        default: kubric-multiview dex_ycb cmupanoptic_3dgs
#
# Examples:
#   bash scripts/tracktail_cached.sh
#   bash scripts/tracktail_cached.sh "dex_ycb cmupanoptic_3dgs"
#   NETWORKS="main_pretrain_e25w9v86 ablation_latent_dim_256_lczwsdx3" bash scripts/tracktail_cached.sh
#   NETWORKS="all" bash scripts/tracktail_cached.sh        every paper network (main + pretrain + 8 ablations)
#
# Results: $TRACKTAIL_OUT_BASE/<network>_canonical_cached/
#   (default $TRACKTAIL_RESULTS/tracktail-outputs/main_finetune_vzkjqrys_canonical_cached).
# Paper protocol: QUERY_OCC=unknown (default; no GT visibility at the query frame), VIS_THRESH=0.1 (default).
# Env overrides: NETWORKS, TRACKTAIL_OUT_BASE, POSE_FILE (default pose3d_cached.npz), DEVICE (default cuda:0),
#   QUERY_OCC, VIS_THRESH.

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${POSE_FILE:-pose3d_cached.npz}
QUERY_OCC=${QUERY_OCC:-unknown}
VIS_THRESH=${VIS_THRESH:-0.1}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
PYTHON=$PIXI_PYTHON

NETWORKS=${NETWORKS:-main_finetune_vzkjqrys}
if [[ "$NETWORKS" == "all" ]]; then NETWORKS="$PAPER_NETWORKS"; fi
for NAME in $NETWORKS; do model_args "$NAME" >/dev/null || exit 1; done

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

cd "$REPO"
for NAME in $NETWORKS; do
    OUT="$TRACKTAIL_OUT_BASE/${NAME}_canonical_cached"
    echo "starting tracktail eval on cached tracks ($POSE_FILE): $NAME [$(model_args "$NAME")], $DATASETS -> $OUT"

    for DATASET in $DATASETS; do
        $PYTHON scripts/eval_testset_metrics.py \
            $(model_args "$NAME") \
            --root          "$TRACKTAIL_DATA" \
            --out           "$OUT" \
            --datasets      "$DATASET" \
            --view-seed     0 \
            --device        "${DEVICE:-cuda:0}" \
            --pose-file     "$POSE_FILE" \
            --query-occlusion "$QUERY_OCC" \
            --vis-thresh    "$VIS_THRESH" \
            --cam-names     ${CANONICAL_CAMS[$DATASET]}
    done

    # every call above rewrites summary.json with its own dataset only: rebuild the combined one
    $PYTHON scripts/merge_eval_summaries.py "$OUT"
done

echo "done!"
