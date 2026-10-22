#!/usr/bin/env bash
#
# MVTracker on MVTracker's cached tracks, on the canonical cameras of the MVTracker tables, with the
# paper-protocol metrics. Same steps and output layout as mvtracker_100.sh, on pose3d_cached.npz
# (scripts/make_pose3d_cached.py) with --query-first: each track is queried at the frame the file gives it.
# --official-settings gives the model MVTracker's own per-dataset settings (visibility threshold, support grid,
# 4 iterations, scene normalization). Input is 256x256, which the predictor resizes to 384x512.
#
# The canonical DUSt3R depths are linked into the posetail trial dirs first (run precompute_dust3r_native.sh
# canonical before, as for mvtracker_100.sh). The old script, mvtracker_100.sh, is unchanged.
#
# Both depth variants are run by default: DUSt3R depths (all datasets) and ground-truth depths (kubric and
# cmupanoptic_3dgs; dex_ycb has none), read from the dataset's own files. Each goes to its own folder:
#   $OUTPUT_BASE/paper-canonical-cached-duster   and   $OUTPUT_BASE/paper-canonical-cached-gt
#
# Usage:
#   bash mvtracker_cached.sh [DATASETS] [POSE_FILE] [DEPTH_TYPES]
#
# Examples:
#   bash mvtracker_cached.sh
#   bash mvtracker_cached.sh "dex_ycb cmupanoptic_3dgs"
#   bash mvtracker_cached.sh "kubric-multiview cmupanoptic_3dgs" pose3d_cached.npz gt
#
# Env overrides: OUTPUT_BASE, OUTPUT_TAG (default "cached", the middle part of the folder name), DEVICE (default
# cuda:0), PYTHON.

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${2:-pose3d_cached.npz}
DEPTH_TYPES=${3:-"duster gt"}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r:$REPO:$PYTHONPATH"
export TORCH_HOME=/groups/karashchuk/home/ruppk2/.cache/torch

PYTHON="${PYTHON:-/home/ruppk2@hhmi.org/envs/mvtracker-official/bin/python}"
CKPT=/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/mvtracker_200000_june2025.pth
DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
DEPTH_ROOT="$DATASET_ROOT"
OUTPUT_BASE=${OUTPUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/mvtracker-outputs}

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

# Native dataset paths and canonical depth subdir names (from precompute_dust3r_native.sh)
declare -A NATIVE_ROOTS=(
    [kubric-multiview]="/groups/karashchuk/karashchuklab/mvtracker-datasets/kubric-multiview/test"
    [dex_ycb]="/groups/karashchuk/karashchuklab/mvtracker-datasets/dex-ycb-multiview"
    [cmupanoptic_3dgs]="/groups/karashchuk/karashchuklab/mvtracker-datasets/panoptic-multiview"
)
declare -A NATIVE_DEPTH_SUBDIRS=(
    [kubric-multiview]="duster-views-0-1-2-3"
    [dex_ycb]="duster-views-0-1-2-3"
    [cmupanoptic_3dgs]="duster-views-1-7-14-20"
)

# ── Step 1: link native canonical depths into posetail trial dirs ─────────────
# Fast (symlinks only) and idempotent — skips if the link already exists.
echo "=== Step 1: linking canonical depths ==="
for DATASET in $DATASETS; do
    NATIVE_ROOT="${NATIVE_ROOTS[$DATASET]:-}"
    DEPTH_SUBDIR="${NATIVE_DEPTH_SUBDIRS[$DATASET]:-}"
    for SESSION_DIR in "$DATASET_ROOT/$DATASET/test"/*/; do
        SESSION=$(basename "$SESSION_DIR")
        src="$NATIVE_ROOT/$SESSION/$DEPTH_SUBDIR"
        dst="$DATASET_ROOT/$DATASET/test/$SESSION/trial/$DEPTH_SUBDIR"
        if [[ ! -d "$src" ]]; then
            echo "  [warn] native depth not found: $src"
            echo "         Run: bash precompute_dust3r_native.sh canonical"
        elif [[ ! -e "$dst" ]]; then
            ln -s "$src" "$dst"
            echo "  linked: $DATASET/$SESSION/$DEPTH_SUBDIR"
        fi
    done
done

# ── Steps 2-4 for each depth type ─────────────────────────────────────────────
for DEPTH in $DEPTH_TYPES; do
    OUTPUT_ROOT="$OUTPUT_BASE/paper-canonical-${OUTPUT_TAG:-cached}-$DEPTH"

    # datasets this depth type exists for
    RUN_DATASETS=""
    for DATASET in $DATASETS; do
        if [[ "$DEPTH" == "gt" && "$DATASET" == "dex_ycb" ]]; then
            echo "  [skip] no GT depths for dex_ycb"
        else
            RUN_DATASETS="$RUN_DATASETS $DATASET"
        fi
    done

    # ── Step 2: MVTracker inference (each dataset with its own canonical cameras) ──
    echo "=== Step 2: MVTracker inference ($DEPTH depth) -> $OUTPUT_ROOT ==="
    for DATASET in $RUN_DATASETS; do
        if [[ "$DEPTH" == "gt" ]]; then
            DEPTH_ARGS="--depth-type gt-native --native-root ${NATIVE_ROOTS[$DATASET]}"
        else
            DEPTH_ARGS="--depth-type duster"
        fi
        $PYTHON "$SCRIPTS/mvtracker_inference.py" \
            --dataset-root "$DATASET_ROOT" \
            --depth-root   "$DEPTH_ROOT" \
            --output-root  "$OUTPUT_ROOT" \
            --datasets "$DATASET" \
            --split test \
            --checkpoint "$CKPT" \
            --device "$DEVICE" \
            --cam-names ${CANONICAL_CAMS[$DATASET]} \
            --view-seed 0 \
            --pose-file "$POSE_FILE" \
            --query-first \
            --official-settings \
            $DEPTH_ARGS
    done

    # ── Step 3: compute metrics (paper protocol) ──────────────────────────────
    echo "=== Step 3: metrics ($DEPTH depth) ==="
    $PYTHON "$SCRIPTS/mvtracker_metrics.py" \
        --dataset-root "$DATASET_ROOT" \
        --output-root  "$OUTPUT_ROOT" \
        --datasets $RUN_DATASETS \
        --split test \
        --force

    # ── Step 4: combine per-dataset metrics into summary ─────────────────────
    echo "=== Step 4: combine metrics ($DEPTH depth) ==="
    (
        cd "$REPO"
        if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then CPY="$REPO/.pixi/envs/default/bin/python"; else CPY="pixi run python"; fi
        $CPY scripts/combine_metrics.py --prefix "$OUTPUT_ROOT"
    )
done
