#!/usr/bin/env bash
# Tracktail n-camera ablation on dex_ycb (n_views = 1 … 8), see eval_ncam_ablation_local.sh.
# Each N value is submitted as an independent bsub job running eval_ncam_ablation_local.sh for that N (this repo has
# to be visible on the cluster nodes). Combine afterwards with combine_ncam_ablation.py.

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS=/groups/karashchuk/home/ruppk2/results/posetail-ablations-logs
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs/ncam_ablation_dexycb}
export OUT_BASE

mkdir -p "$LOGS"
for N in $(seq 1 8); do
    bsub -J "eval_ncam_n${N}" -W 24:00 -e "$LOGS/%J.err" -o "$LOGS/%J.out" \
        -n 4 -q gpu_a100 -gpu "num=1" \
        bash "$REPO/scripts/eval_ncam_ablation_local.sh" "$N"
done

echo ""
echo "=== All jobs submitted. Run combine_ncam_ablation.py after they finish: ==="
echo "pixi run python $REPO/cotracker_baseline/combine_ncam_ablation.py $OUT_BASE"
