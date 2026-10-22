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

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/home/ruppk2@hhmi.org/software/co-tracker:$SCRIPTS/..:$PYTHONPATH"
cd "$SCRIPTS"
if [[ -x "$SCRIPTS/../.pixi/envs/default/bin/python" ]]; then PYTHON="$SCRIPTS/../.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

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

DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
OUTPUT_ROOT=/groups/karashchuk/home/ruppk2/results/paper-final-v2/cotracker-outputs/paper-${NCAMS_LABEL}cams-cached-100
CKPT=/home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth

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
            --device cuda:0 \
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
        --device cuda:0 \
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
$PYTHON /home/ruppk2@hhmi.org/posetail/scripts/combine_metrics.py \
    --prefix $OUTPUT_ROOT
