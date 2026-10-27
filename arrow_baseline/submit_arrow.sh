#!/usr/bin/env bash
# Submit the ARROW evaluation to the cluster (LSF, one GPU per job, L4 by default), run from a machine with ssh to login2 or from
# a login node itself. One job per dataset (and per camera count on Point Odyssey), each running arrow_cached.sh /
# arrow_cached_po.sh for that piece; every job scores its datasets and re-combines the metrics of its output folder,
# so the last job to finish leaves the complete combined_*.csv.
#
# Usage:
#   bash submit_arrow.sh [WHICH] [DATASETS]   WHICH = all (default) | cached | po | po4 | po9 | 100; DATASETS limits the datasets
#   e.g. bash submit_arrow.sh po9 point-odyssey-human
#
# Env overrides: QUEUE (default gpu_l4), WALL (default 24:00), LOGS, plus everything arrow_cached*.sh read
# (OUTPUT_ROOT / OUTPUT_BASE, SCALE_MODE, WINDOW, MAX_ANCHORS, CKPT, ARROW_PYTHON), which are passed to the jobs.
#
# Setup (once): ARROW's environment has to live on /groups so the nodes see it:
#   ssh login2 'cd /groups/karashchuk/home/karashchukl/projects/posetail/arrow && \
#     UV_PYTHON_INSTALL_DIR=/groups/karashchuk/home/karashchukl/.local/share/uv/python \
#     UV_CACHE_DIR=/groups/karashchuk/home/karashchukl/scratch/uv-cache UV_PROJECT_ENVIRONMENT=.venv-cluster \
#     uv sync --python 3.13'
# and the weights are a local copy of the ifradlin/arrow hub files in /groups/karashchuk/home/karashchukl/weights/arrow.

set -e

WHICH=${1:-all}
ONLY=${2:-}
wanted() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }
QUEUE=${QUEUE:-gpu_l4}   # peak ~15 GB allocated / ~20 GB reserved (9 cams, 132 images) fits an L4 (24 GB)
WALL=${WALL:-24:00}
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # physical path: the nodes do not see ~/ghome
LOGS=${LOGS:-/groups/karashchuk/home/karashchukl/logs/arrow}
mkdir -p "$LOGS"

# environment passed to the jobs (only the variables that are set)
ENVS=""
for V in OUTPUT_ROOT OUTPUT_BASE SCALE_MODE WINDOW MAX_ANCHORS CKPT ARROW_PYTHON; do
    [[ -n "${!V}" ]] && ENVS="$ENVS $V=${!V}"
done

submit() {   # name, command
    local cmd="cd $SCRIPTS && env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True$ENVS $2"
    if command -v bsub >/dev/null 2>&1; then
        bsub -J "$1" -W "$WALL" -n 8 -q "$QUEUE" -gpu "num=1" -e "$LOGS/$1-%J.err" -o "$LOGS/$1-%J.out" "$cmd"
    else
        ssh -n -o BatchMode=yes login2 "bsub -J '$1' -W '$WALL' -n 8 -q '$QUEUE' -gpu 'num=1' -e '$LOGS/$1-%J.err' -o '$LOGS/$1-%J.out' \"$cmd\""
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
echo "logs: $LOGS"
