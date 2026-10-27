#!/usr/bin/env bash
# Tracktail n-camera ablation on dex_ycb. Runs directly (no bsub), same style as tracktail_cached.sh.
#
# N = 1 ... 8 cameras, nested: view_00 ... view_{N-1} (N = 4 is the canonical set of the MVTracker tables). Every N
# runs on its own track file, pose3d_ncam{N}_512.npz (scripts/make_pose3d_ncam.py), which the CoTracker and MVTracker
# ablations use too: 512 tracks seen by at least min(2, N) of the cameras, queried at the first frame they are seen in.
# Paper-protocol metrics (eval_testset_metrics.py; --legacy-metrics there for the previous pooled definition).
#
# Usage:
#   bash eval_ncam_ablation_local.sh [N ...]      default: 1 2 3 4 5 6 7 8
#
# Results: $OUT_BASE/nviews-N/, and ncam_ablation_summary.csv over all N at the end.
# Env overrides: OUT_BASE, DEVICE (default cuda:0).

set -e

NS=${*:-"1 2 3 4 5 6 7 8"}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f ~/.setup_env_vars.sh ]] && source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

WANDB_RUN=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb/run-20260912_115732-vzkjqrys
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs/ncam_ablation_dexycb}

cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

for N in $NS; do
    CAMS=$(for i in $(seq 0 $((N-1))); do printf "view_%02d " $i; done)
    echo ""
    echo "=== n_views=${N} (${CAMS}) ==="
    $PYTHON scripts/eval_testset_metrics.py \
        --wandb-folder "$WANDB_RUN" \
        --out           "$OUT_BASE/nviews-${N}" \
        --datasets      dex_ycb \
        --view-seed     0 \
        --device        "${DEVICE:-cuda:0}" \
        --pose-file     "pose3d_ncam${N}_512.npz" \
        --cam-names     $CAMS
done

echo ""
echo "=== Combining all n_views ==="
$PYTHON cotracker_baseline/combine_ncam_ablation.py "$OUT_BASE"
