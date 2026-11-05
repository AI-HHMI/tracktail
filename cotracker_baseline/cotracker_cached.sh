#!/usr/bin/env bash
# CoTracker3 baseline on MVTracker's cached tracks, on the canonical cameras of the MVTracker tables, with the
# paper-protocol metrics. Same steps as cotracker_100.sh, on pose3d_cached.npz (scripts/make_pose3d_cached.py) with
# --query-first: each track is queried at the frame the file gives it, not at frame 0.
#
# Usage:
#   bash cotracker_cached.sh [DATASETS] [POSE_FILE]
#
# Examples:
#   bash cotracker_cached.sh
#   bash cotracker_cached.sh "dex_ycb cmupanoptic_3dgs"
#
# Results: $OUTPUT_ROOT (default $COTRACKER_OUT_BASE/paper-canonical-cached).
# Paths come from paper.env (TRACKTAIL_DATA, COTRACKER_REPO, COTRACKER_CKPT); runs in the pixi env.
# Env overrides: DATASET_ROOT, OUTPUT_ROOT, DEVICE (default cuda:0), PYTHON, CKPT.

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${2:-pose3d_cached.npz}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$COTRACKER_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$PIXI_PYTHON}
CKPT=${CKPT:-$COTRACKER_CKPT}
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
OUTPUT_ROOT=${OUTPUT_ROOT:-$COTRACKER_OUT_BASE/paper-canonical-cached}
cd "$SCRIPTS"

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

echo "starting cotracker eval on cached tracks ($POSE_FILE): $DATASETS -> $OUTPUT_ROOT"

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
    --query-first

# triangulate per dataset, each with its own canonical camera subset
for DATASET in $DATASETS; do
    $PYTHON cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASET \
        --split test \
        --cam-names ${CANONICAL_CAMS[$DATASET]} \
        --view-seed 0 \
        --pose-file "$POSE_FILE"
done

# compute the evaluation metrics (paper protocol)
$PYTHON cotracker_metrics.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --force \
    --pose-file "$POSE_FILE"

# combine the evaluation metrics into a summary dataframe
$PYTHON "$REPO/scripts/combine_metrics.py" --prefix $OUTPUT_ROOT

echo "done!"
