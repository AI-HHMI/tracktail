#!/usr/bin/env bash
# ARROW (RGB-only, https://github.com/ifradlin/arrow) on the curated 512 tracks of the Point Odyssey test sets
# (pose3d.npz, scripts/make_pose3d_512.py), with the paper-protocol metrics. Same steps as arrow_cached.sh and the same
# tracks / cameras / query frames as cotracker_cached_po.sh and mvtracker_cached_po.sh: --query-first, each track is
# queried at its first frame visible in the cameras that are used.
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Both are run by
# default, each into its own folder (the query frames depend on the cameras):
#   $OUTPUT_BASE/paper-po-4cams   and   $OUTPUT_BASE/paper-po-9cams
# The 499 frames are run in windows of WINDOW timesteps (see arrow_inference.py).
#
# Usage:
#   bash arrow_cached_po.sh [CAMS] [DATASETS] [POSE_FILE]
#
# Examples:
#   bash arrow_cached_po.sh
#   bash arrow_cached_po.sh 4
#   bash arrow_cached_po.sh "4 9" point-odyssey-human
#
# Env overrides: DATASET_ROOT, OUTPUT_BASE, DEVICE (default cuda:0), SCALE_MODE (gt-tracks | cameras, default
# gt-tracks), WINDOW (timesteps per encoding, default 24 for 4 cams, 12 for 9 cams), MAX_ANCHORS (default 24),
# ARROW_PYTHON (default ../arrow/.venv-cluster or .venv), CKPT.

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
POSE_FILE=${3:-pose3d.npz}
DEVICE=${DEVICE:-cuda:0}
SCALE_MODE=${SCALE_MODE:-gt-tracks}
MAX_ANCHORS=${MAX_ANCHORS:-24}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
# ARROW's uv environment: .venv-cluster (built on login2, python on /groups, see submit_arrow.sh) when it exists
if [[ -z "$ARROW_PYTHON" ]]; then
    if [[ -x "$REPO/../arrow/.venv-cluster/bin/python" ]]; then ARROW_PYTHON="$REPO/../arrow/.venv-cluster/bin/python"
    else ARROW_PYTHON="$REPO/../arrow/.venv/bin/python"; fi   # cd ../arrow && uv sync
fi
CKPT=${CKPT:-/groups/karashchuk/home/karashchukl/weights/arrow}   # local copy of the ifradlin/arrow hub weights

DATASET_ROOT=${DATASET_ROOT:-/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs}

cd "$SCRIPTS"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
        N_WINDOW=${WINDOW:-24}
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
        N_WINDOW=${WINDOW:-12}
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi
    OUTPUT_ROOT="$OUTPUT_BASE/paper-po-${N}cams"
    echo "starting arrow eval on the 512 Point Odyssey tracks ($N cameras): $DATASETS -> $OUTPUT_ROOT"

    # run ARROW inference (all cameras jointly)
    $ARROW_PYTHON arrow_inference.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        --device $DEVICE \
        --checkpoint "$CKPT" \
        --pose-file "$POSE_FILE" \
        --query-first \
        --scale-mode $SCALE_MODE \
        --window $N_WINDOW \
        --max-anchors $MAX_ANCHORS \
        $CAM_ARG

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
done
