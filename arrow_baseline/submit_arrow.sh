#!/usr/bin/env bash
# Run the ARROW evaluation in pieces: one piece per dataset (and per camera count on Point Odyssey), each running
# arrow_cached.sh / arrow_cached_po.sh / arrow_100.sh for that piece; every piece scores its datasets and re-combines
# the metrics of its output folder, so the last one to finish leaves the complete combined_*.csv.
# The pieces run here, one after the other.
#
# Usage:
#   bash submit_arrow.sh [WHICH] [DATASETS]   WHICH = all (default) | cached | po | po4 | po9 | 100; DATASETS limits the datasets
#   e.g. bash submit_arrow.sh po9 point-odyssey-human
#
# Env overrides: everything arrow_cached*.sh read (OUTPUT_ROOT / OUTPUT_BASE, DATASET_ROOT, SCALE_MODE, WINDOW,
# MAX_ANCHORS, CKPT, ARROW_PYTHON), which are passed to the pieces.
# GPU memory: an L4 (24 GB) fits everything except po9 at the default WINDOW=24, which needs ~A100-80GB or more.
#
# Setup (once): ARROW's uv env and weights (baselines/setup.sh; ARROW_PYTHON, ARROW_CKPT in paper.env).

set -e

WHICH=${1:-all}
ONLY=${2:-}
wanted() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPTS/../paper.env"

# environment passed to the jobs (only the variables that are set)
ENVS=""
for V in OUTPUT_ROOT OUTPUT_BASE DATASET_ROOT SCALE_MODE WINDOW MAX_ANCHORS CKPT ARROW_PYTHON; do
    [[ -n "${!V}" ]] && ENVS="$ENVS $V=${!V}"
done

submit() {   # name, command: run here
    local cmd="cd $SCRIPTS && env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True$ENVS $2"
    echo "=== $1 ==="
    bash -c "$cmd"
}

if [[ "$WHICH" == "all" || "$WHICH" == "cached" ]]; then
    for DATASET in kubric-multiview dex_ycb cmupanoptic_3dgs; do
        wanted $DATASET || continue
        submit "arrow-cached-$DATASET" "bash arrow_cached.sh $DATASET"
    done
fi
if [[ "$WHICH" == "100" ]]; then   # arrow_100.sh (100-track subsets for the renders), one job per dataset
    for DATASET in kubric-multiview dex_ycb cmupanoptic_3dgs; do
        wanted $DATASET || continue
        submit "arrow-100-$DATASET" "bash arrow_100.sh canonical $DATASET"
    done
fi
if [[ "$WHICH" == "all" || "$WHICH" == po* ]]; then
    NS="4 9"; [[ "$WHICH" == "po4" ]] && NS=4; [[ "$WHICH" == "po9" ]] && NS=9
    for N in $NS; do
        for DATASET in po-animal-sub256-motion point-odyssey-human; do
            wanted $DATASET || continue
            submit "arrow-po${N}-$DATASET" "bash arrow_cached_po.sh $N $DATASET"
        done
    done
fi
