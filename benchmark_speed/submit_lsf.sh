#!/usr/bin/env bash
# Submit the speed benchmark (all methods, sequentially on one A100) to LSF. Run from a login node, e.g.
#   ssh login2 "cd $PWD && bash benchmark_speed/submit_lsf.sh"
#
# Usage: bash benchmark_speed/submit_lsf.sh [METHODS] [QUEUE]
#   METHODS default: all ("tracktail cotracker mvtracker-gt mvtracker-dust3r arrow")
#   QUEUE   default: gpu_a100 (e.g. gpu_l4: 24 GB GPUs, out-of-memory configurations are recorded as "oom")
# Env: SLOTS (CPU slots, default 12), WALL (default 12:00), OUT_DIR, BENCH_ARGS, PER_METHOD.
# One job per method runs in parallel on separate GPUs if PER_METHOD=1 (each still gets a whole GPU to itself).

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS=${1:-"tracktail cotracker mvtracker-gt mvtracker-dust3r arrow"}
QUEUE=${2:-gpu_a100}
WALL=${WALL:-12:00}
OUT_DIR=${OUT_DIR:-"$HERE/results/${QUEUE}_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUT_DIR/lsf"

submit() {   # name, methods
    bsub -J "bench-$1" -q "$QUEUE" -n "${SLOTS:-12}" -gpu "num=1:mode=exclusive_process" -R "span[hosts=1]" -W "$WALL" \
        -o "$OUT_DIR/lsf/$1.%J.out" -e "$OUT_DIR/lsf/$1.%J.err" \
        "BENCH_ARGS='${BENCH_ARGS:-}' bash $HERE/run_all.sh $OUT_DIR '$2'"
}

if [[ "${PER_METHOD:-0}" == "1" ]]; then
    for M in $METHODS; do submit "$M" "$M"; done
else
    submit all "$METHODS"
fi
echo "results -> $OUT_DIR"
