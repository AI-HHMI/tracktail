#!/usr/bin/env bash
# CoTracker3 baseline on 100 tracks per trial: a random subset of MVTracker's cached tracks (pose3d_cached_100.npz,
# scripts/make_pose3d_cached_100.py), with the paper-protocol metrics (cotracker_metrics.py; --legacy-metrics there for
# the previous pooled definition) and --query-first (each track is queried at the frame the file gives it).
#
# bash cotracker_100.sh [canonical|N_VIEWS] [DATASETS] [POSE_FILE]
# e.g. bash cotracker_100.sh canonical
#      bash cotracker_100.sh canonical "dex_ycb cmupanoptic_3dgs"
# canonical: the cameras of the MVTracker tables. N_VIEWS (random cameras) is not meaningful for the cached tracks,
# whose visibility is only given for the canonical cameras.
# Results: $OUTPUT_BASE/paper-<canonical|N>cams-cached-100 (default OUTPUT_BASE = $COTRACKER_OUT_BASE).
# Paths come from paper.env (TRACKTAIL_DATA, COTRACKER_REPO, COTRACKER_CKPT); runs in the pixi env.
# Env overrides: DATASET_ROOT, OUTPUT_BASE, OUTPUT_ROOT, DEVICE (default cuda:0), PYTHON, CKPT.

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$COTRACKER_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$PIXI_PYTHON}
CKPT=${CKPT:-$COTRACKER_CKPT}
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
DEVICE=${DEVICE:-cuda:0}
cd "$SCRIPTS"

MODE=${1:-canonical}
DATASETS=${2:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${3:-pose3d_cached_100.npz}

if [[ "$MODE" == "canonical" ]]; then
    NCAMS_LABEL="canonical"
    N_VIEWS_ARG=""
else
    NCAMS_LABEL="$MODE"
    N_VIEWS_ARG="--n-views $MODE"
fi

OUTPUT_BASE=${OUTPUT_BASE:-$COTRACKER_OUT_BASE}
OUTPUT_ROOT=${OUTPUT_ROOT:-$OUTPUT_BASE/paper-${NCAMS_LABEL}cams-cached-100}

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

# run inference on cotracker baseline (predict individually on each camera view): per dataset when canonical, so each
# runs only its own cameras and the query frames count visibility in those cameras
if [[ "$MODE" == "canonical" ]]; then
    for DATASET in $DATASETS; do
        $PYTHON cotracker_inference.py \
            --dataset-root $DATASET_ROOT \
            --output-root $OUTPUT_ROOT \
            --datasets $DATASET \
            --split test \
            --checkpoint $CKPT \
            --device "$DEVICE" \
            --query-batch-size 1024 \
            --pose-file "$POSE_FILE" \
            --query-first \
            --cam-names ${CANONICAL_CAMS[$DATASET]}
    done
else
    $PYTHON cotracker_inference.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        --checkpoint $CKPT \
        --device "$DEVICE" \
        --query-batch-size 1024 \
        --pose-file "$POSE_FILE" \
        --query-first
fi

# triangulate — per-dataset when canonical so each gets its own camera subset
if [[ "$MODE" == "canonical" ]]; then
    for DATASET in $DATASETS; do
        CAM_ARG=""
        if [[ -n "${CANONICAL_CAMS[$DATASET]}" ]]; then
            CAM_ARG="--cam-names ${CANONICAL_CAMS[$DATASET]}"
        fi
        $PYTHON cotracker_triangulate.py \
            --dataset-root $DATASET_ROOT \
            --output-root $OUTPUT_ROOT \
            --datasets $DATASET \
            --split test \
            $CAM_ARG \
            --view-seed 0 \
            --pose-file "$POSE_FILE"
    done
else
    $PYTHON cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        $N_VIEWS_ARG \
        --view-seed 0 \
        --pose-file "$POSE_FILE"
fi

# compute the evaluation metrics
$PYTHON cotracker_metrics.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --force \
    --pose-file "$POSE_FILE"

# combine the evaluation metrics into a summary dataframe
$PYTHON "$REPO/scripts/combine_metrics.py" \
    --prefix $OUTPUT_ROOT
