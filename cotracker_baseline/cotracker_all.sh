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
# Results: $OUTPUT_BASE/paper-<canonical|Ncams|all> (default OUTPUT_BASE = $COTRACKER_OUT_BASE). Not used for the
# paper tables (see cotracker_cached.sh, cotracker_cached_po.sh).
# Paths come from paper.env (TRACKTAIL_DATA, COTRACKER_REPO, COTRACKER_CKPT); runs in the pixi env.
# Env overrides: DATASETS, DATASET_ROOT, OUTPUT_BASE, DEVICE (default cuda:0), PYTHON, CKPT.

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
export PYTHONPATH="$COTRACKER_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-$PIXI_PYTHON}
CKPT=${CKPT:-$COTRACKER_CKPT}
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
DEVICE=${DEVICE:-cuda:0}
cd "$SCRIPTS"

MODE=${1:-}      # number of cameras, "canonical", or empty (all)
POSE_FILE=${2:-pose3d.npz}
# e.g. DATASETS="kubric-multiview dex_ycb cmupanoptic_3dgs tapvid3d_adt tapvid3d_drivetrack tapvid3d_pstudio"
DATASETS=${DATASETS:-"po-animal-sub256-motion point-odyssey-human"}

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

OUTPUT_ROOT=${OUTPUT_BASE:-$COTRACKER_OUT_BASE}/paper-${NCAMS_LABEL}

# run inference on cotracker baseline (predict individually on each camera view)
# inference always runs on all cameras — view selection happens at triangulation
$PYTHON cotracker_inference.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --checkpoint $CKPT \
    --device "$DEVICE" \
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
        $PYTHON cotracker_triangulate.py \
            --dataset-root $DATASET_ROOT \
            --output-root $OUTPUT_ROOT \
            --datasets $DATASET \
            --split test \
            $CAM_ARG \
            --view-seed 0
    done
else
    $PYTHON cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root $OUTPUT_ROOT \
        --datasets $DATASETS \
        --split test \
        $N_VIEWS_ARG \
        --view-seed 0
fi

# compute the evaluation metrics
$PYTHON cotracker_metrics.py \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --datasets $DATASETS \
    --split test \
    --force

# combine the evaluation metrics into a summary dataframe
$PYTHON "$REPO/scripts/combine_metrics.py" \
    --prefix $OUTPUT_ROOT
