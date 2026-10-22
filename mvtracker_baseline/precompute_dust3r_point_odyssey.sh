#!/usr/bin/env bash
# Precompute DUSt3R depths for Point Odyssey datasets.
# Outputs are written into the dataset directories under posetail-pretraining-v6.
#
# Usage:
#   bash precompute_dust3r_point_odyssey.sh [N_VIEWS]
#   N_VIEWS: number of cameras to subsample (default: all)
#
# Examples:
#   bash precompute_dust3r_point_odyssey.sh      # all cams
#   bash precompute_dust3r_point_odyssey.sh 4   # 4 cams with seed 0

set -e

N_VIEWS=${1:-}

cd /groups/karashchuk/home/ruppk2

export PYTHONPATH="/groups/karashchuk/home/ruppk2/github/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r:/groups/karashchuk/home/ruppk2/github/posetail:$PYTHONPATH"

PYTHON=/groups/karashchuk/home/ruppk2/miniforge3/envs/mvtracker/bin/python
SCRIPT=/groups/karashchuk/home/ruppk2/github/posetail/mvtracker_baseline/precompute_dust3r_depths.py
CKPT=/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6

N_VIEWS_ARG=${N_VIEWS:+--n-views $N_VIEWS}

for DATASET in point-odyssey-human po-animal-sub256-motion; do
    echo "=== $DATASET ==="
    $PYTHON $SCRIPT \
        --dataset-root "$DATASET_ROOT" \
        --depth-root   "$DATASET_ROOT" \
        --datasets     "$DATASET" \
        --split test \
        --checkpoint "$CKPT" \
        --device cuda:0 \
        $N_VIEWS_ARG \
        --view-seed 0
done
