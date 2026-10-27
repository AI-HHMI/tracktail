#!/usr/bin/env bash
# ARROW on 100 tracks per trial: a random subset of MVTracker's cached tracks (pose3d_cached_100.npz,
# scripts/make_pose3d_cached_100.py), with the paper-protocol metrics and --query-first (each track is queried at the
# frame the file gives it). Same runs as cotracker_100.sh / tracktail_100.sh; the outputs are what arrow_render.sh
# renders for the qualitative figure.
#
# Usage:
#   bash arrow_100.sh [canonical] [DATASETS] [POSE_FILE]
#   canonical: the cameras of the MVTracker tables (the only mode: random cameras are not meaningful for the cached
#   tracks, whose visibility is only given for the canonical cameras).
#
# Examples:
#   bash arrow_100.sh
#   bash arrow_100.sh canonical "dex_ycb cmupanoptic_3dgs"
#
# Env overrides: OUTPUT_ROOT, DEVICE (default cuda:0), SCALE_MODE (gt-tracks | cameras, default gt-tracks), WINDOW
# (default 24), MAX_ANCHORS (default 24), ARROW_PYTHON (default ../arrow/.venv-cluster or .venv), CKPT.

set -e

MODE=${1:-canonical}
DATASETS=${2:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${3:-pose3d_cached_100.npz}
DEVICE=${DEVICE:-cuda:0}
SCALE_MODE=${SCALE_MODE:-gt-tracks}
WINDOW=${WINDOW:-24}
MAX_ANCHORS=${MAX_ANCHORS:-24}

if [[ "$MODE" != "canonical" ]]; then
    echo "only the canonical cameras are supported (the cached tracks have visibility for those only), got $MODE"
    exit 1
fi

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
if [[ -z "$ARROW_PYTHON" ]]; then
    if [[ -x "$REPO/../arrow/.venv-cluster/bin/python" ]]; then ARROW_PYTHON="$REPO/../arrow/.venv-cluster/bin/python"
    else ARROW_PYTHON="$REPO/../arrow/.venv/bin/python"; fi   # cd ../arrow && uv sync
fi
CKPT=${CKPT:-/groups/karashchuk/home/karashchukl/weights/arrow}   # local copy of the ifradlin/arrow hub weights

DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
OUTPUT_ROOT=${OUTPUT_ROOT:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/paper-canonicalcams-cached-100}

cd "$SCRIPTS"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

echo "starting arrow 100-point eval ($POSE_FILE): $DATASETS -> $OUTPUT_ROOT"

# run ARROW inference (all cameras jointly), per dataset so each runs its own canonical cameras
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
