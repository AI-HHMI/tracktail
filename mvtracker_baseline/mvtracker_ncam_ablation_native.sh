#!/usr/bin/env bash
# MVTracker n-camera ablation on dex_ycb (n_views = 1 … 8).
#
# Nested cameras view_00 ... view_{N-1} (N = 4 is the canonical set of the MVTracker tables), on the track file
# pose3d_ncam{N}_512.npz of each N (scripts/make_pose3d_ncam.py), which the CoTracker and Tracktail ablations use too:
# 512 tracks seen by at least min(2, N) of the cameras, queried at the first frame they are seen in. Run through
# mvtracker_inference.py like mvtracker_cached.sh: --query-first, MVTracker's official dex_ycb settings and scene
# normalization, 256x256 input, and the paper-protocol metrics (mvtracker_metrics.py).
#
# Unlike CoTracker, the DUSt3R depths depend on which cameras are used together, so they are precomputed per N (skipped
# when the folder exists) in the native dataset and linked into the posetail trial dirs. DUSt3R needs at least 2 views:
# for N = 1 the depth of view_00 is reconstructed from the pair view_00 + view_01 and only view_00 is given to MVTracker.
#
# Usage:
#   bash mvtracker_ncam_ablation_native.sh [N ...]      default: 1 2 3 4 5 6 7 8
#
# Results: $ABLATION_ROOT/nviews-N/, and ncam_ablation_summary.csv over all N at the end.
# Env overrides: ABLATION_ROOT, DEVICE (default cuda:0), PYTHON (default: the mvtracker-official venv).

set -e

NS=${*:-"1 2 3 4 5 6 7 8"}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r:$REPO:$PYTHONPATH"
export TORCH_HOME=/groups/karashchuk/home/ruppk2/.cache/torch

PYTHON="${PYTHON:-/home/ruppk2@hhmi.org/envs/mvtracker-official/bin/python}"
DUST3R_PYTHON=/home/ruppk2@hhmi.org/miniforge3/envs/mvtracker/bin/python      # as in precompute_dust3r_native.sh
DUST3R_PATH=/home/ruppk2@hhmi.org/software/dust3r
DATA_ROOT=/groups/karashchuk/karashchuklab/mvtracker-datasets
NATIVE_ROOT=$DATA_ROOT/dex-ycb-multiview
DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
CKPT_DUSTER=$DATA_ROOT/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
CKPT_MVTRACKER=$DATA_ROOT/checkpoints/mvtracker_200000_june2025.pth
ABLATION_ROOT=${ABLATION_ROOT:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/mvtracker-outputs/ncam-ablation-dexycb}

for N in $NS; do
    OUT_ROOT="${ABLATION_ROOT}/nviews-${N}"
    POSE_FILE="pose3d_ncam${N}_512.npz"
    CAMS=$(for i in $(seq 0 $((N-1))); do printf "view_%02d " $i; done)
    if [ "$N" -eq 1 ]; then DUST_VIEWS="0 1"; else DUST_VIEWS=$(seq -s ' ' 0 $((N-1))); fi
    DEPTH_SUBDIR="duster-views-$(echo $DUST_VIEWS | tr ' ' '-')"

    echo ""
    echo "=== n_views=${N} (${CAMS}), depth ${DEPTH_SUBDIR} ==="

    echo "-- DUSt3R precompute (skips if already done) --"
    PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:$DUST3R_PATH:$PYTHONPATH" $DUST3R_PYTHON "$SCRIPTS/precompute_dust3r_native.py" \
        --dataset-type dex-ycb-multiview \
        --data-root  "$DATA_ROOT" \
        --checkpoint "$CKPT_DUSTER" \
        --device "$DEVICE" \
        --views $DUST_VIEWS

    echo "-- linking the depths into the posetail trial dirs --"
    for SESSION_DIR in "$DATASET_ROOT/dex_ycb/test"/*/; do
        SESSION=$(basename "$SESSION_DIR")
        src="$NATIVE_ROOT/$SESSION/$DEPTH_SUBDIR"
        dst="$DATASET_ROOT/dex_ycb/test/$SESSION/trial/$DEPTH_SUBDIR"
        if [[ ! -d "$src" ]]; then
            echo "  [warn] native depth not found: $src"
        elif [[ ! -e "$dst" ]]; then
            ln -s "$src" "$dst"
        fi
    done

    echo "-- MVTracker inference --"
    $PYTHON "$SCRIPTS/mvtracker_inference.py" \
        --dataset-root "$DATASET_ROOT" \
        --depth-root   "$DATASET_ROOT" \
        --output-root  "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --checkpoint "$CKPT_MVTRACKER" \
        --device "$DEVICE" \
        --cam-names $CAMS \
        --view-seed 0 \
        --pose-file "$POSE_FILE" \
        --query-first \
        --official-settings \
        --depth-type duster \
        --depth-subdir "$DEPTH_SUBDIR"

    echo "-- Metrics --"
    $PYTHON "$SCRIPTS/mvtracker_metrics.py" \
        --dataset-root "$DATASET_ROOT" \
        --output-root  "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --force

    echo "-- Combining metrics --"
    $PYTHON "$REPO/scripts/combine_metrics.py" --prefix "$OUT_ROOT"
done

# ── Combine per-nviews summaries into a single CSV ───────────────────────────
echo ""
echo "=== Combining all n_views ==="
$PYTHON "$REPO/cotracker_baseline/combine_ncam_ablation.py" "$ABLATION_ROOT"
