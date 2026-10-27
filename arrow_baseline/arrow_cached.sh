#!/usr/bin/env bash
# ARROW (RGB-only, https://github.com/ifradlin/arrow) on MVTracker's cached tracks, on the canonical cameras of the
# MVTracker tables, with the paper-protocol metrics. Same tracks, cameras and query frames as cotracker_cached.sh and
# mvtracker_cached.sh: pose3d_cached.npz (scripts/make_pose3d_cached.py) with --query-first, each track queried at the
# frame the file gives it, as a 2D pixel in a camera that sees it (see arrow_inference.py for the protocol).
#
# Usage:
#   bash arrow_cached.sh [DATASETS] [POSE_FILE]
#
# Examples:
#   bash arrow_cached.sh
#   bash arrow_cached.sh "dex_ycb cmupanoptic_3dgs"
#
# Env overrides: OUTPUT_ROOT, DEVICE (default cuda:0), SCALE_MODE (gt-tracks | cameras, default gt-tracks),
# WINDOW (timesteps per encoding, default 24), MAX_ANCHORS (default 24), ARROW_PYTHON (default ../arrow/.venv-cluster or .venv), CKPT.

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${2:-pose3d_cached.npz}
DEVICE=${DEVICE:-cuda:0}
SCALE_MODE=${SCALE_MODE:-gt-tracks}
WINDOW=${WINDOW:-24}
MAX_ANCHORS=${MAX_ANCHORS:-24}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
# ARROW's uv environment: .venv-cluster (built on login2, python on /groups, see submit_arrow.sh) when it exists
if [[ -z "$ARROW_PYTHON" ]]; then
    if [[ -x "$REPO/../arrow/.venv-cluster/bin/python" ]]; then ARROW_PYTHON="$REPO/../arrow/.venv-cluster/bin/python"
    else ARROW_PYTHON="$REPO/../arrow/.venv/bin/python"; fi   # cd ../arrow && uv sync
fi
CKPT=${CKPT:-/groups/karashchuk/home/karashchukl/weights/arrow}   # local copy of the ifradlin/arrow hub weights

DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
OUTPUT_ROOT=${OUTPUT_ROOT:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/paper-canonical-cached}

cd "$SCRIPTS"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

echo "starting arrow eval on cached tracks ($POSE_FILE): $DATASETS -> $OUTPUT_ROOT"

# run ARROW inference (all cameras jointly, each dataset with its own canonical cameras)
for DATASET in $DATASETS; do
    $ARROW_PYTHON arrow_inference.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASET \
        --split test \
        --device $DEVICE \
        --checkpoint "$CKPT" \
        --cam-names ${CANONICAL_CAMS[$DATASET]} \
        --pose-file "$POSE_FILE" \
        --query-first \
        --scale-mode $SCALE_MODE \
        --window $WINDOW \
        --max-anchors $MAX_ANCHORS
done

# compute the evaluation metrics (paper protocol)
$PYTHON "$REPO/mvtracker_baseline/mvtracker_metrics.py" \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --prefix arrow_ \
    --force

# combine the evaluation metrics into a summary dataframe
$PYTHON "$REPO/scripts/combine_metrics.py" --prefix $OUTPUT_ROOT

echo "done!"
