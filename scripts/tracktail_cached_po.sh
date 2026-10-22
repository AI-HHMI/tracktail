#!/usr/bin/env bash
#
# Tracktail inference + metrics on the curated 512 tracks of the Point Odyssey test sets (pose3d.npz,
# scripts/make_pose3d_512.py). Runs directly (no bsub), same style as tracktail_cached.sh.
#   scripts/eval_testset_metrics.py   inference (query_first: each track at the first frame visible in the cameras that
#                                     are used) and the metrics (the paper protocol by default; add --legacy-metrics
#                                     there for the previous pooled definition)
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Both are run by
# default, each into its own folder:
#   $OUT_BASE/main_finetune_vzkjqrys_po_4cams   and   $OUT_BASE/main_finetune_vzkjqrys_po_9cams
#
# Usage:
#   bash tracktail_cached_po.sh [CAMS] [DATASETS]
#
# Examples:
#   bash tracktail_cached_po.sh
#   bash tracktail_cached_po.sh 4
#   bash tracktail_cached_po.sh "4 9" point-odyssey-human
#
# Env overrides: OUT_BASE, POSE_FILE (default pose3d.npz), DEVICE (default cuda:0).

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
POSE_FILE=${POSE_FILE:-pose3d.npz}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

WANDB_BASE=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs}

cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi
    OUT="$OUT_BASE/main_finetune_vzkjqrys_po_${N}cams"
    echo "starting tracktail eval on the 512 Point Odyssey tracks ($N cameras, $POSE_FILE): $DATASETS -> $OUT"

    # both datasets use the same cameras, so one call writes one summary.json / metrics.md over both
    $PYTHON scripts/eval_testset_metrics.py \
        --wandb-folder "$WANDB_BASE/run-20260912_115732-vzkjqrys" \
        --out           "$OUT" \
        --datasets      $DATASETS \
        --view-seed     0 \
        --device        "${DEVICE:-cuda:0}" \
        --pose-file     "$POSE_FILE" \
        $CAM_ARG
done

echo "done!"
