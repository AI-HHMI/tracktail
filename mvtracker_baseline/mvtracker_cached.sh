#!/usr/bin/env bash
#
# MVTracker on MVTracker's cached tracks, on the canonical cameras of the MVTracker tables, with the
# paper-protocol metrics. Same steps and output layout as mvtracker_100.sh, on pose3d_cached.npz
# (scripts/make_pose3d_cached.py) with --query-first: each track is queried at the frame the file gives it.
# --official-settings gives the model MVTracker's own per-dataset settings (visibility threshold, support grid,
# 4 iterations, scene normalization). Input is 256x256, which the predictor resizes to 384x512.
#
# The canonical DUSt3R depths are linked from MVTracker's native datasets ($MVTRACKER_DATA) into the trial dirs
# first (run precompute_dust3r_native.sh canonical before, as for mvtracker_100.sh).
#
# Both depth variants are run by default: DUSt3R depths and ground-truth depths, read from the native datasets
# ($MVTRACKER_DATA): kubric and cmupanoptic_3dgs use their rendered/reconstructed depth, dex_ycb its Kinect sensor
# depth (--depth-resize linear, the same call as mvtracker_dexycb_sensordepth.sh). Each goes to its own folder:
#   $OUTPUT_BASE/paper-canonical-cached-duster   and   $OUTPUT_BASE/paper-canonical-cached-gt
# (default OUTPUT_BASE = $MVTRACKER_OUT_BASE = $TRACKTAIL_RESULTS/mvtracker-outputs).
# Paper Tab. main_results: MVTracker DUSt3R rows and GT rows for Kubric/Panoptic come from this script. The DexYCB
# GT (†Kinect) number was produced by mvtracker_dexycb_sensordepth.sh (separate folder), with identical settings.
#
# Usage:
#   bash mvtracker_cached.sh [DATASETS] [POSE_FILE] [DEPTH_TYPES]
#
# Examples:
#   bash mvtracker_cached.sh
#   bash mvtracker_cached.sh "dex_ycb cmupanoptic_3dgs"
#   bash mvtracker_cached.sh "kubric-multiview cmupanoptic_3dgs" pose3d_cached.npz gt
#
# Paths come from paper.env (TRACKTAIL_DATA, MVTRACKER_DATA, MVTRACKER_REPO, DUST3R_REPO, MVT_HUB_DIR, MVTRACKER_CKPT,
# MVTRACKER_PYTHON). Env overrides: DATASET_ROOT, OUTPUT_BASE, OUTPUT_TAG (default "cached", the middle part of the
# folder name), DEVICE (default cuda:0), PYTHON, CKPT.

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${2:-pose3d_cached.npz}
DEPTH_TYPES=${3:-"duster gt"}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
BASELINE_LD=${LD_LIBRARY_PATH:-}   # LD_LIBRARY_PATH without the pixi libs of paper_common.sh (those are for the pixi python)
while [[ "$BASELINE_LD" == "$PAPER_REPO/.pixi/envs/default/lib:"* ]]; do BASELINE_LD=${BASELINE_LD#"$PAPER_REPO/.pixi/envs/default/lib:"}; done
export PYTHONPATH="$MVTRACKER_REPO:$DUST3R_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MVT_HUB_DIR                        # torch.hub loads the pinned clone (paper.env)

PYTHON=${PYTHON:-env LD_LIBRARY_PATH=$BASELINE_LD $MVTRACKER_PYTHON}
CKPT=${CKPT:-$MVTRACKER_CKPT}
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
DEPTH_ROOT="$DATASET_ROOT"
OUTPUT_BASE=${OUTPUT_BASE:-$MVTRACKER_OUT_BASE}

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

# Native dataset paths and canonical depth subdir names (from precompute_dust3r_native.sh)
MVTRACKER_DATA=${MVTRACKER_DATA:?set MVTRACKER_DATA (paper.env)}
declare -A NATIVE_ROOTS=(
    [kubric-multiview]="$MVTRACKER_DATA/kubric-multiview/test"
    [dex_ycb]="$MVTRACKER_DATA/dex-ycb-multiview"
    [cmupanoptic_3dgs]="$MVTRACKER_DATA/panoptic-multiview"
)
declare -A NATIVE_DEPTH_SUBDIRS=(
    [kubric-multiview]="duster-views-0-1-2-3"
    [dex_ycb]="duster-views-0-1-2-3"
    [cmupanoptic_3dgs]="duster-views-1-7-14-20"
)

# ── Step 1: link native canonical depths into the trial dirs ─────────────────
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
    RUN_DATASETS="$DATASETS"

    # ── Step 2: MVTracker inference (each dataset with its own canonical cameras) ──
    echo "=== Step 2: MVTracker inference ($DEPTH depth) -> $OUTPUT_ROOT ==="
    for DATASET in $RUN_DATASETS; do
        if [[ "$DEPTH" == "gt" && "$DATASET" == "dex_ycb" ]]; then
            # Kinect sensor depth (view_XX/depth/*.png, mm -> m), resized bilinearly (= mvtracker_dexycb_sensordepth.sh)
            DEPTH_ARGS="--depth-type gt-native --native-root ${NATIVE_ROOTS[$DATASET]} --depth-resize linear"
        elif [[ "$DEPTH" == "gt" ]]; then
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
    (cd "$REPO" && $PIXI_PYTHON scripts/combine_metrics.py --prefix "$OUTPUT_ROOT")
done
