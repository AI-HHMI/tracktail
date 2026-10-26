#!/usr/bin/env bash
# ARROW n-camera ablation on dex_ycb, matched to Tracktail / CoTracker / MVTracker.
#
# For each N=1..8 this uses nested cameras view_00 ... view_{N-1} and the shared curated file
# pose3d_ncam{N}_512.npz (scripts/make_pose3d_ncam.py): the same 512 sampled tracks and query frames used by the other
# n-camera ablations. ARROW receives RGB only. Each 3D query is projected into a visible source camera; predicted
# positions are scored with the paper-protocol metric script, after per-encoding scale fitting from GT tracks (the
# default scale mode used for the ARROW cached runs). No depth or calibration is passed to the model.
#
# Usage:
#   bash arrow_ncam_ablation.sh [N ...]       default: 1 2 3 4 5 6 7 8
#   bash arrow_ncam_ablation.sh 1 4 8         run selected camera counts
#
# Results: $ABLATION_ROOT/nviews-N/ and $ABLATION_ROOT/ncam_ablation_summary.csv.
# Env overrides: ABLATION_ROOT, DATASET_ROOT, DEVICE (default cuda:0), SCALE_MODE (gt-tracks | cameras), WINDOW,
# MAX_ANCHORS, ARROW_PYTHON, CKPT. On LSF, submit with e.g.:
#   bsub -J arrow-ncam -W 24:00 -n 8 -q gpu_l4 -gpu 'num=1' \
#       -o "$HOME/logs/arrow/ncam-%J.out" -e "$HOME/logs/arrow/ncam-%J.err" \
#       bash arrow_ncam_ablation.sh

set -euo pipefail

NS=("$@")
if [[ ${#NS[@]} -eq 0 ]]; then NS=(1 2 3 4 5 6 7 8); fi

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPTS/.." && pwd -P)"
DATASET_ROOT=${DATASET_ROOT:-/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6}
ABLATION_ROOT=${ABLATION_ROOT:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs/ncam-ablation-dexycb}
DEVICE=${DEVICE:-cuda:0}
SCALE_MODE=${SCALE_MODE:-gt-tracks}
WINDOW=${WINDOW:-24}
MAX_ANCHORS=${MAX_ANCHORS:-24}
QUERY_SEED=${QUERY_SEED:-0}

if [[ -z "${ARROW_PYTHON:-}" ]]; then
    if [[ -x "$REPO/../arrow/.venv-cluster/bin/python" ]]; then
        ARROW_PYTHON="$REPO/../arrow/.venv-cluster/bin/python"
    else
        ARROW_PYTHON="$REPO/../arrow/.venv/bin/python"
    fi
fi
CKPT=${CKPT:-/groups/karashchuk/home/karashchukl/weights/arrow}
if [[ ! -x "$ARROW_PYTHON" ]]; then
    echo "ARROW Python not executable: $ARROW_PYTHON (see arrow_baseline/submit_arrow.sh for cluster setup)" >&2
    exit 1
fi

if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then
    METRICS_PYTHON=("$REPO/.pixi/envs/default/bin/python")
elif command -v pixi >/dev/null 2>&1; then
    METRICS_PYTHON=(pixi run python)
else
    echo "Cannot find the posetail pixi Python for mvtracker_metrics.py" >&2
    exit 1
fi

mkdir -p "$ABLATION_ROOT"
cd "$SCRIPTS"
for N in "${NS[@]}"; do
    if ! [[ "$N" =~ ^[1-8]$ ]]; then
        echo "N must be one of 1..8, got '$N'" >&2
        exit 1
    fi
    CAMS=()
    for ((i=0; i<N; i++)); do CAMS+=("view_$(printf '%02d' "$i")"); done
    POSE_FILE="pose3d_ncam${N}_512.npz"
    OUT_ROOT="$ABLATION_ROOT/nviews-${N}"
    echo ""
    echo "=== ARROW n_views=$N (${CAMS[*]}), pose=$POSE_FILE -> $OUT_ROOT ==="

    "$ARROW_PYTHON" arrow_inference.py \
        --dataset-root "$DATASET_ROOT" \
        --output-root "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --checkpoint "$CKPT" \
        --device "$DEVICE" \
        --cam-names "${CAMS[@]}" \
        --pose-file "$POSE_FILE" \
        --query-first \
        --query-seed "$QUERY_SEED" \
        --scale-mode "$SCALE_MODE" \
        --window "$WINDOW" \
        --max-anchors "$MAX_ANCHORS"

    "${METRICS_PYTHON[@]}" "$REPO/mvtracker_baseline/mvtracker_metrics.py" \
        --dataset-root "$DATASET_ROOT" \
        --output-root "$OUT_ROOT" \
        --datasets dex_ycb \
        --split test \
        --prefix arrow_ \
        --force

    "${METRICS_PYTHON[@]}" "$REPO/scripts/combine_metrics.py" --prefix "$OUT_ROOT"
done

echo ""
echo "=== Combining n_views summaries ==="
"${METRICS_PYTHON[@]}" "$REPO/cotracker_baseline/combine_ncam_ablation.py" "$ABLATION_ROOT"
