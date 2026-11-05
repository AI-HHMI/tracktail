#!/usr/bin/env bash
# ARROW (RGB-only, https://github.com/ifradlin/arrow) on the curated 512 tracks of the Point Odyssey test sets
# (pose3d.npz, scripts/make_pose3d_512.py), with the paper-protocol metrics. Same steps as arrow_cached.sh and the same
# tracks / cameras / query frames as cotracker_cached_po.sh and mvtracker_cached_po.sh: --query-first, each track is
# queried at its first frame visible in the cameras that are used.
#
# CAMS is the number of cameras: 4 = views 2, 4, 5, 7 (view2 view4 view5 view7), 9 = all cameras. Both are run by
# default, each into its own folder (the query frames depend on the cameras):
#   $OUTPUT_BASE/paper-po-4cams   and   $OUTPUT_BASE/paper-po-9cams   (default OUTPUT_BASE = $ARROW_OUT_BASE)
# The 499 frames are run in windows of WINDOW timesteps (see arrow_inference.py), 24 by default as in the paper.
# 9 cams x 24 frames needs a large GPU (~A100-80GB or more). The original 9-cam runs used WINDOW=12 (to fit an L4);
# Tab. po_results' 9-view ARROW numbers are the WINDOW=24 rerun, which is what the defaults here produce
# ($ARROW_OUT_BASE/paper-po-9cams; tracktail-paper reads that folder).
#
# Usage:
#   bash arrow_cached_po.sh [CAMS] [DATASETS] [POSE_FILE]
#
# Examples:
#   bash arrow_cached_po.sh
#   bash arrow_cached_po.sh 4
#   bash arrow_cached_po.sh "4 9" point-odyssey-human
#
# Paths come from paper.env (TRACKTAIL_DATA, ARROW_CKPT, ARROW_PYTHON; ARROW's uv env: cd $ARROW_REPO && uv sync).
# Env overrides: DATASET_ROOT, OUTPUT_BASE, DEVICE (default cuda:0), SCALE_MODE (gt-tracks | cameras, default
# gt-tracks), WINDOW (timesteps per encoding, default 24), MAX_ANCHORS (default 24), ARROW_PYTHON, CKPT.

set -e

CAMS=${1:-"4 9"}
DATASETS=${2:-"po-animal-sub256-motion point-odyssey-human"}
POSE_FILE=${3:-pose3d.npz}
DEVICE=${DEVICE:-cuda:0}
SCALE_MODE=${SCALE_MODE:-gt-tracks}
MAX_ANCHORS=${MAX_ANCHORS:-24}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
BASELINE_LD=${LD_LIBRARY_PATH:-}   # LD_LIBRARY_PATH without the pixi libs of paper_common.sh (those are for the pixi python)
while [[ "$BASELINE_LD" == "$PAPER_REPO/.pixi/envs/default/lib:"* ]]; do BASELINE_LD=${BASELINE_LD#"$PAPER_REPO/.pixi/envs/default/lib:"}; done
CKPT=${CKPT:-$ARROW_CKPT}             # local copy of the ifradlin/arrow hub weights (baselines/setup.sh)
PYTHON=$PIXI_PYTHON                   # metrics + combine (ARROW itself runs in $ARROW_PYTHON)
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
OUTPUT_BASE=${OUTPUT_BASE:-$ARROW_OUT_BASE}
cd "$SCRIPTS"

for N in $CAMS; do
    if [[ "$N" == "4" ]]; then
        CAM_ARG="--cam-names view2 view4 view5 view7"
        N_WINDOW=${WINDOW:-24}
    elif [[ "$N" == "9" ]]; then
        CAM_ARG=""
        N_WINDOW=${WINDOW:-24}   # needs ~A100-80GB; WINDOW=12 fits an L4
    else
        echo "CAMS must be 4 and/or 9, got $N"; exit 1
    fi
    OUTPUT_ROOT="$OUTPUT_BASE/paper-po-${N}cams"
    echo "starting arrow eval on the 512 Point Odyssey tracks ($N cameras): $DATASETS -> $OUTPUT_ROOT"

    # run ARROW inference (all cameras jointly)
    env LD_LIBRARY_PATH="$BASELINE_LD" "$ARROW_PYTHON" arrow_inference.py \
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
