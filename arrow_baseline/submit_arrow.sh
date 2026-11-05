#!/usr/bin/env bash
# Run the ARROW evaluation in pieces: one piece per dataset (and per camera count on Point Odyssey), each running
# arrow_cached.sh / arrow_cached_po.sh / arrow_100.sh for that piece; every piece scores its datasets and re-combines
# the metrics of its output folder, so the last one to finish leaves the complete combined_*.csv.
# By default the pieces run here, one after the other. Optional: SUBMIT=1 submits each piece as an LSF job (bsub, one
# GPU per job) instead.
#
# Usage:
#   bash submit_arrow.sh [WHICH] [DATASETS]   WHICH = all (default) | cached | po | po4 | po9 | 100; DATASETS limits the datasets
#   e.g. bash submit_arrow.sh po9 point-odyssey-human
#
# Env overrides: everything arrow_cached*.sh read (OUTPUT_ROOT / OUTPUT_BASE, DATASET_ROOT, SCALE_MODE, WINDOW,
# MAX_ANCHORS, CKPT, ARROW_PYTHON), which are passed to the pieces; with SUBMIT=1 also QUEUE (default gpu_l4),
# WALL (default 24:00), LOGS (default $TRACKTAIL_LOGS/arrow).
# GPU memory: an L4 (24 GB) fits everything except po9 at the default WINDOW=24, which needs ~A100-80GB or more
# (e.g. SUBMIT=1 QUEUE=gpu_h100 bash submit_arrow.sh po9).
#
# Setup (once): ARROW's uv env and weights (baselines/setup.sh; ARROW_PYTHON, ARROW_CKPT in paper.env). With SUBMIT=1
# they have to be on a filesystem the cluster nodes see.

set -e

WHICH=${1:-all}
ONLY=${2:-}
wanted() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }
QUEUE=${QUEUE:-gpu_l4}   # peak ~15 GB allocated / ~20 GB reserved (9 cams, 132 images) fits an L4 (24 GB)
WALL=${WALL:-24:00}
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # physical path (cluster nodes may not see symlinked homes)
source "$SCRIPTS/../paper.env"
LOGS=${LOGS:-$TRACKTAIL_LOGS/arrow}
if [[ -n "$SUBMIT" ]]; then
    command -v bsub >/dev/null 2>&1 || { echo "SUBMIT=1 needs LSF (bsub)" >&2; exit 1; }
    mkdir -p "$LOGS"
fi

# environment passed to the jobs (only the variables that are set)
ENVS=""
for V in OUTPUT_ROOT OUTPUT_BASE DATASET_ROOT SCALE_MODE WINDOW MAX_ANCHORS CKPT ARROW_PYTHON; do
    [[ -n "${!V}" ]] && ENVS="$ENVS $V=${!V}"
done

submit() {   # name, command: run here, or with SUBMIT=1 as an LSF job
    local cmd="cd $SCRIPTS && env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True$ENVS $2"
    if [[ -n "$SUBMIT" ]]; then
        bsub -J "$1" -W "$WALL" -n 8 -q "$QUEUE" -gpu "num=1" -e "$LOGS/$1-%J.err" -o "$LOGS/$1-%J.out" "$cmd"
    else
        echo "=== $1 ==="
        bash -c "$cmd"
    fi
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
if [[ -n "$SUBMIT" ]]; then echo "logs: $LOGS"; fi
