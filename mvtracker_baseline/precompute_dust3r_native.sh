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
#
# Depths are written into MVTracker's native datasets under $MVTRACKER_DATA (<scene>/duster-views-<views>/).
# Paths come from paper.env (MVTRACKER_DATA, MVTRACKER_REPO, DUST3R_REPO, DUST3R_CKPT, DUST3R_PYTHON).
# Env overrides: PYTHON, CKPT, DEVICE (default cuda:0).

set -e

MODE=${1:-}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
BASELINE_LD=${LD_LIBRARY_PATH:-}   # LD_LIBRARY_PATH without the pixi libs of paper_common.sh (those are for the pixi python)
while [[ "$BASELINE_LD" == "$PAPER_REPO/.pixi/envs/default/lib:"* ]]; do BASELINE_LD=${BASELINE_LD#"$PAPER_REPO/.pixi/envs/default/lib:"}; done
export PYTHONPATH="$MVTRACKER_REPO:$DUST3R_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-env LD_LIBRARY_PATH=$BASELINE_LD $DUST3R_PYTHON}
CKPT=${CKPT:-$DUST3R_CKPT}
DEVICE=${DEVICE:-cuda:0}
SCRIPT="$SCRIPTS/precompute_dust3r_native.py"
DATA_ROOT=${MVTRACKER_DATA:?set MVTRACKER_DATA (paper.env)}

if [[ "$MODE" == "canonical" ]]; then
    VIEWS_ARG="--mvtracker-canonical"
elif [[ -n "$MODE" ]]; then
    VIEWS_ARG="--n-views $MODE"
else
    VIEWS_ARG=""
fi

for DATASET in kubric-multiview dex-ycb-multiview panoptic-multiview; do
    echo "=== $DATASET ==="
    $PYTHON "$SCRIPT" \
        --dataset-type "$DATASET" \
        --data-root  "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --device "$DEVICE" \
        $VIEWS_ARG \
        --view-seed 0
done
