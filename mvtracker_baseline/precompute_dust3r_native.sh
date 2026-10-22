#!/usr/bin/env bash
# Precompute DUSt3R depths for MVTracker-native datasets (kubric, dex-ycb, panoptic).
#
# Usage:
#   bash precompute_dust3r_native.sh [N_VIEWS|canonical]
#   N_VIEWS:   number of cameras to subsample with seed 0 (default: all)
#   canonical: use the exact camera subsets from MVTracker's main evaluation table
#
# Examples:
#   bash precompute_dust3r_native.sh           # all cams, all datasets
#   bash precompute_dust3r_native.sh 4         # 4 cams with seed 0, all datasets
#   bash precompute_dust3r_native.sh canonical # MVTracker main-table cameras

set -e

MODE=${1:-}

cd ~

export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:/home/ruppk2@hhmi.org/software/dust3r:$PYTHONPATH"

PYTHON=/home/ruppk2@hhmi.org/miniforge3/envs/mvtracker/bin/python
SCRIPT=/home/ruppk2@hhmi.org/posetail/mvtracker_baseline/precompute_dust3r_native.py
CKPT=/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
DATA_ROOT=/groups/karashchuk/karashchuklab/mvtracker-datasets

if [[ "$MODE" == "canonical" ]]; then
    VIEWS_ARG="--mvtracker-canonical"
elif [[ -n "$MODE" ]]; then
    VIEWS_ARG="--n-views $MODE"
else
    VIEWS_ARG=""
fi

for DATASET in kubric-multiview dex-ycb-multiview panoptic-multiview; do
    echo "=== $DATASET ==="
    $PYTHON $SCRIPT \
        --dataset-type "$DATASET" \
        --data-root  "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --device cuda:0 \
        $VIEWS_ARG \
        --view-seed 0
done
