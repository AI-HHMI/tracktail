#!/usr/bin/env bash
#
# Tracktail inference + metrics on 100 tracks per trial: a random subset of MVTracker's cached tracks
# (pose3d_cached_100.npz, scripts/make_pose3d_cached_100.py). query_first: each track is queried at the frame the file
# gives it; the metrics are the paper protocol (eval_testset_metrics.py; --legacy-metrics there for the previous pooled
# definition). Runs directly (no bsub), same style as cotracker_100.sh / mvtracker_100.sh.
#
# Usage:
#   bash tracktail_100.sh [canonical|N_VIEWS] [DATASETS] [POSE_FILE]
#   canonical: the cameras of the MVTracker tables. N_VIEWS (random cameras) is not meaningful for the cached tracks,
#   whose visibility is only given for the canonical cameras.
#
# Examples:
#   bash tracktail_100.sh
#   bash tracktail_100.sh canonical "dex_ycb cmupanoptic_3dgs"

set -e

MODE=${1:-canonical}
DATASETS=${2:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${3:-pose3d_cached_100.npz}

if [[ "$MODE" == "canonical" ]]; then
    NCAMS_LABEL="canonical"
else
    NCAMS_LABEL="${MODE}cams"
fi

source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

WANDB_BASE=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb
OUT=/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs/main_finetune_vzkjqrys_${NCAMS_LABEL}_cached_100

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

echo "starting tracktail 100-point eval"

if [[ "$MODE" == "canonical" ]]; then
    for DATASET in $DATASETS; do
        CAM_ARG=""
        if [[ -n "${CANONICAL_CAMS[$DATASET]}" ]]; then
            CAM_ARG="--cam-names ${CANONICAL_CAMS[$DATASET]}"
        fi
        $PYTHON scripts/eval_testset_metrics.py \
            --wandb-folder "$WANDB_BASE/run-20260912_115732-vzkjqrys" \
            --out           "$OUT" \
            --datasets      "$DATASET" \
            --view-seed     0 \
            --pose-file     "$POSE_FILE" \
            $CAM_ARG
    done
else
    $PYTHON scripts/eval_testset_metrics.py \
        --wandb-folder "$WANDB_BASE/run-20260912_115732-vzkjqrys" \
        --out           "$OUT" \
        --datasets      $DATASETS \
        --n-views       "$MODE" \
        --view-seed     0 \
        --pose-file     "$POSE_FILE"
fi

if [[ "$MODE" == "canonical" ]]; then
    # every call above rewrites summary.json with its own dataset only: rebuild the combined one
    $PYTHON scripts/merge_eval_summaries.py "$OUT"
fi

echo "done!"
