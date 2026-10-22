#!/usr/bin/env bash
#
# MVTracker on the curated 512 tracks of the Point Odyssey test sets (pose3d.npz, scripts/make_pose3d_512.py), with
# the paper-protocol metrics. Same steps and output layout as mvtracker_cached.sh, with --query-first: each track is
# queried at its first frame visible in the cameras that are used. --official-settings gives the model MVTracker's
# settings (visibility threshold 0.01, support grid 4, 4 iterations; Point Odyssey has no scene normalization).
# Input is 256x256, which the predictor resizes to 384x512.
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Depth is the DUSt3R
# depths and the ground-truth depths. Everything is run by default, each combination into its own folder:
#   $OUTPUT_BASE/paper-po-<CAMS>cams-<DEPTH>      e.g. paper-po-4cams-duster, paper-po-9cams-gt
# The depth folders (duster-views-2-4-5-7, duster-views-0-1-2-3-4-5-6-7-8, gt-depths) are already in the trial dirs.
#
# Usage:
#   bash mvtracker_cached_po.sh [CAMS] [DATASETS] [DEPTH_TYPES] [POSE_FILE]
#
# Examples:
#   bash mvtracker_cached_po.sh
#   bash mvtracker_cached_po.sh 4
#   bash mvtracker_cached_po.sh "4 9" point-odyssey-human gt
#
# VIS_THRESHOLD replaces MVTracker's visibility threshold (default: the setting of the dataset in mvtracker_inference.py,
# 0.01 for Point Odyssey) and, unless OUTPUT_SUFFIX is given, appends it to the output folders (paper-po-4cams-duster_0.5)
# so earlier results are kept. The probability of being visible is saved as vis_prob in every output file.
#
# Env overrides: DATASET_ROOT, OUTPUT_BASE, VIS_THRESHOLD, OUTPUT_SUFFIX, DEVICE (default cuda:0), PYTHON, QUERY_CHUNK (queries per model call,
# default: all 512 at once; lower it if the GPU runs out of memory).

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
DEPTH_TYPES=${3:-"duster gt"}
POSE_FILE=${4:-pose3d.npz}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r:$REPO:$PYTHONPATH"
export TORCH_HOME=/groups/karashchuk/home/ruppk2/.cache/torch

PYTHON="${PYTHON:-/home/ruppk2@hhmi.org/envs/mvtracker-official/bin/python}"
CKPT=/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/mvtracker_200000_june2025.pth
DATASET_ROOT=${DATASET_ROOT:-/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6}
DEPTH_ROOT="$DATASET_ROOT"
OUTPUT_BASE=${OUTPUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/mvtracker-outputs}

if [[ -n "$VIS_THRESHOLD" && -z "${OUTPUT_SUFFIX+x}" ]]; then OUTPUT_SUFFIX="_$VIS_THRESHOLD"; fi
VIS_ARG=${VIS_THRESHOLD:+--vis-threshold $VIS_THRESHOLD}

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi

    for DEPTH in $DEPTH_TYPES; do
        OUTPUT_ROOT="$OUTPUT_BASE/paper-po-${N}cams-${DEPTH}${OUTPUT_SUFFIX}"
        echo "=== MVTracker on the 512 Point Odyssey tracks ($N cameras, $DEPTH depth): $DATASETS -> $OUTPUT_ROOT ==="

        # inference (the depth subdir is derived from the cameras: duster-views-2-4-5-7 / duster-views-0-1-...-8, gt-depths)
        $PYTHON "$SCRIPTS/mvtracker_inference.py" \
            --dataset-root "$DATASET_ROOT" \
            --depth-root   "$DEPTH_ROOT" \
            --output-root  "$OUTPUT_ROOT" \
            --datasets $DATASETS \
            --split test \
            --checkpoint "$CKPT" \
            --device "$DEVICE" \
            --depth-type "$DEPTH" \
            $CAM_ARG \
            --view-seed 0 \
            --pose-file "$POSE_FILE" \
            --query-first \
            --official-settings \
            $VIS_ARG \
            ${QUERY_CHUNK:+--query-chunk-size $QUERY_CHUNK}

        # metrics (paper protocol)
        $PYTHON "$SCRIPTS/mvtracker_metrics.py" \
            --dataset-root "$DATASET_ROOT" \
            --output-root  "$OUTPUT_ROOT" \
            --datasets $DATASETS \
            --split test \
            --force

        # combine per-dataset metrics into a summary
        (
            cd "$REPO"
            if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then CPY="$REPO/.pixi/envs/default/bin/python"; else CPY="pixi run python"; fi
            $CPY scripts/combine_metrics.py --prefix "$OUTPUT_ROOT"
        )
    done
done
