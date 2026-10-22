#!/usr/bin/env bash
# CoTracker n-camera ablation on dex_ycb (n_views = 2 … 8; triangulation needs at least 2 cameras).
#
# Nested cameras view_00 ... view_{N-1} (N = 4 is the canonical set of the MVTracker tables), on the track file
# pose3d_ncam{N}_512.npz of each N (scripts/make_pose3d_ncam.py), which the Tracktail and MVTracker ablations use too:
# 512 tracks seen by at least 2 of the cameras, queried at the first frame they are seen in. The tracks and query
# frames depend on N, so inference runs once per N (on its cameras only), then triangulation and the paper-protocol
# metrics (cotracker_metrics.py; --legacy-metrics there for the previous pooled definition).
#
# Usage:
#   bash cotracker_ncam_ablation.sh [N ...]      default: 2 3 4 5 6 7 8
#
# Results: $ABLATION_ROOT/nviews-N/, and ncam_ablation_summary.csv over all N at the end.
# Env overrides: ABLATION_ROOT, DEVICE (default cuda:0).

set -e

NS=${*:-"2 3 4 5 6 7 8"}
DEVICE=${DEVICE:-cuda:0}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPTS/.."
export PYTHONPATH="/home/ruppk2@hhmi.org/software/co-tracker:$REPO:$PYTHONPATH"

DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
CKPT=/home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth
ABLATION_ROOT=${ABLATION_ROOT:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/cotracker-outputs/ncam-ablation-dexycb}

cd "$SCRIPTS"
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then PYTHON="$REPO/.pixi/envs/default/bin/python"; else PYTHON="pixi run python"; fi

for N in $NS; do
    OUT_ROOT="${ABLATION_ROOT}/nviews-${N}"
    POSE_FILE="pose3d_ncam${N}_512.npz"
    CAMS=$(for i in $(seq 0 $((N-1))); do printf "view_%02d " $i; done)

    echo "=== n_views=${N} (${CAMS}) ==="

    $PYTHON cotracker_inference.py \
        --dataset-root $DATASET_ROOT \
        --output-root  "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --checkpoint $CKPT \
        --device $DEVICE \
        --query-batch-size 1024 \
        --pose-file "$POSE_FILE" \
        --query-first \
        --cam-names $CAMS

    $PYTHON cotracker_triangulate.py \
        --dataset-root $DATASET_ROOT \
        --output-root  "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --cam-names $CAMS \
        --view-seed 0 \
        --pose-file "$POSE_FILE" \
        --overwrite

    $PYTHON cotracker_metrics.py \
        --dataset-root $DATASET_ROOT \
        --output-root  "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --pose-file "$POSE_FILE" \
        --force

    $PYTHON "$REPO/scripts/combine_metrics.py" --prefix "$OUT_ROOT"
done

echo ""
echo "=== Combining all n_views ==="
$PYTHON "$SCRIPTS/combine_ncam_ablation.py" "$ABLATION_ROOT"
