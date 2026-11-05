#!/usr/bin/env bash
# Precompute DUSt3R depths for Point Odyssey datasets.
# Outputs are written into the trial dirs under $TRACKTAIL_DATA (<trial>/duster-views-<views>/).
# Paths come from paper.env (TRACKTAIL_DATA, MVTRACKER_REPO, DUST3R_REPO, DUST3R_CKPT, DUST3R_PYTHON).
# Env overrides: DATASET_ROOT, PYTHON, CKPT, DEVICE (default cuda:0).
#
# Usage:
#   bash precompute_dust3r_point_odyssey.sh [N_VIEWS]
#   N_VIEWS: 4 = the paper's 4 cameras view2 view4 view5 view7 (duster-views-2-4-5-7); empty = all 9 cameras
#   (duster-views-0-1-2-3-4-5-6-7-8); other numbers subsample cameras with seed 0.
# The paper needs both: bash precompute_dust3r_point_odyssey.sh 4 && bash precompute_dust3r_point_odyssey.sh

set -e

N_VIEWS=${1:-}

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
BASELINE_LD=${LD_LIBRARY_PATH:-}   # LD_LIBRARY_PATH without the pixi libs of paper_common.sh (those are for the pixi python)
while [[ "$BASELINE_LD" == "$PAPER_REPO/.pixi/envs/default/lib:"* ]]; do BASELINE_LD=${BASELINE_LD#"$PAPER_REPO/.pixi/envs/default/lib:"}; done
export PYTHONPATH="$MVTRACKER_REPO:$DUST3R_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-env LD_LIBRARY_PATH=$BASELINE_LD $DUST3R_PYTHON}
CKPT=${CKPT:-$DUST3R_CKPT}
DEVICE=${DEVICE:-cuda:0}
SCRIPT="$SCRIPTS/precompute_dust3r_depths.py"
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}

if [[ "$N_VIEWS" == "4" ]]; then N_VIEWS_ARG="--cam-names view2 view4 view5 view7"; else N_VIEWS_ARG=${N_VIEWS:+--n-views $N_VIEWS}; fi

for DATASET in point-odyssey-human po-animal-sub256-motion; do
    echo "=== $DATASET ==="
    $PYTHON "$SCRIPT" \
        --dataset-root "$DATASET_ROOT" \
        --depth-root   "$DATASET_ROOT" \
        --datasets     "$DATASET" \
        --split test \
        --checkpoint "$CKPT" \
        --device "$DEVICE" \
        $N_VIEWS_ARG \
        --view-seed 0
done
