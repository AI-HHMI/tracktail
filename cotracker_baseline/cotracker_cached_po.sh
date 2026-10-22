#!/usr/bin/env bash
# CoTracker3 baseline on the curated 512 tracks of the Point Odyssey test sets (pose3d.npz, scripts/make_pose3d_512.py),
# with the paper-protocol metrics. Same steps as cotracker_cached.sh, with --query-first: each track is queried at
# its first frame visible in the cameras that are used.
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Both are run by
# default, each into its own folder (the query frames depend on the cameras):
#   $OUTPUT_BASE/paper-po-4cams   and   $OUTPUT_BASE/paper-po-9cams
#
# Usage:
#   bash cotracker_cached_po.sh [CAMS] [DATASETS] [POSE_FILE]
#
# Examples:
#   bash cotracker_cached_po.sh
#   bash cotracker_cached_po.sh 4
#   bash cotracker_cached_po.sh "4 9" point-odyssey-human
#
# Env overrides: DATASET_ROOT, OUTPUT_BASE, DEVICE (default cuda:0).

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
POSE_FILE=${3:-pose3d.npz}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
export PYTHONPATH="/home/ruppk2@hhmi.org/software/co-tracker:$REPO:$PYTHONPATH"

DATASET_ROOT=${DATASET_ROOT:-/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/cotracker-outputs}
CKPT=/home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth

cd "$SCRIPTS"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi
    OUTPUT_ROOT="$OUTPUT_BASE/paper-po-${N}cams"
    echo "starting cotracker eval on the 512 Point Odyssey tracks ($N cameras): $DATASETS -> $OUTPUT_ROOT"

    # run inference on cotracker baseline (predict individually on each camera view)
    $PYTHON cotracker_inference.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        --checkpoint $CKPT \
        --device $DEVICE \
        --query-batch-size 1024 \
        --pose-file "$POSE_FILE" \
        --query-first \
        $CAM_ARG

    # triangulate the cameras that were run
    $PYTHON cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        --view-seed 0 \
        --pose-file "$POSE_FILE" \
        $CAM_ARG

    # compute the evaluation metrics (paper protocol)
    $PYTHON cotracker_metrics.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        --pose-file "$POSE_FILE" \
        --force

    # combine the evaluation metrics into a summary dataframe
    $PYTHON "$REPO/scripts/combine_metrics.py" --prefix $OUTPUT_ROOT
done
