#!/usr/bin/env bash
#
# Usage:
#   bash cotracker_all.sh [N_VIEWS|canonical] [POSE_FILE]
#   N_VIEWS:   number of cameras to subsample with seed 0 (default: all)
#   canonical: use the exact camera subsets from MVTracker's main evaluation table
#              kubric-multiview  → view_0 view_1 view_2 view_3
#              dex_ycb           → view_00 view_01 view_02 view_03
#              cmupanoptic_3dgs  → 1 7 14 20
#              other datasets    → all cameras

export PYTHONPATH="/home/ruppk2@hhmi.org/software/co-tracker:$PYTHONPATH"

MODE=${1:-}      # number of cameras, "canonical", or empty (all)
POSE_FILE=${2:-pose3d.npz}
DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
CKPT=/home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth
# DATASETS="kubric-multiview dex_ycb cmupanoptic_3dgs tapvid3d_adt tapvid3d_drivetrack tapvid3d_pstudio"
DATASETS="po-animal-sub256-motion point-odyssey-human"

if [[ "$MODE" == "canonical" ]]; then
    NCAMS_LABEL="canonical"
    N_VIEWS_ARG=""
elif [[ -n "$MODE" ]]; then
    NCAMS_LABEL="${MODE}cams"
    N_VIEWS_ARG="--n-views $MODE"
else
    NCAMS_LABEL="all"
    N_VIEWS_ARG=""
fi

OUTPUT_ROOT=/groups/karashchuk/home/ruppk2/results/paper-final-v2/cotracker-outputs/paper-${NCAMS_LABEL}

# run inference on cotracker baseline (predict individually on each camera view)
# inference always runs on all cameras — view selection happens at triangulation
pixi run python cotracker_inference.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --checkpoint $CKPT \
    --device cuda:0 \
    --query-batch-size 1024 \
    --pose-file "$POSE_FILE"

# triangulate — per-dataset when canonical so each gets its own camera subset
if [[ "$MODE" == "canonical" ]]; then
    declare -A CANONICAL_CAMS=(
        [kubric-multiview]="view_0 view_1 view_2 view_3"
        [dex_ycb]="view_00 view_01 view_02 view_03"
        [cmupanoptic_3dgs]="1 7 14 20"
    )
    for DATASET in $DATASETS; do
        CAM_ARG=""
        if [[ -n "${CANONICAL_CAMS[$DATASET]}" ]]; then
            CAM_ARG="--cam-names ${CANONICAL_CAMS[$DATASET]}"
        fi
        pixi run python cotracker_triangulate.py \
            --dataset-root $DATASET_ROOT \
            --output-root $OUTPUT_ROOT \
            --datasets $DATASET \
            --split test \
            $CAM_ARG \
            --view-seed 0
    done
else
    pixi run python cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        $N_VIEWS_ARG \
        --view-seed 0
fi

# compute the evaluation metrics
pixi run python cotracker_metrics.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --force

# combine the evaluation metrics into a summary dataframe
pixi run python /home/ruppk2@hhmi.org/posetail/scripts/combine_metrics.py \
    --prefix $OUTPUT_ROOT
