#!/usr/bin/env bash
#
# Tracktail inference + metrics on MVTracker's cached tracks (pose3d_cached.npz, written by make_pose3d_cached.py),
# on the canonical cameras of the MVTracker tables. Runs directly (no bsub), same style as tracktail_100.sh.
#   1. scripts/eval_testset_metrics.py        inference (query_first: each track at the frame given by the file) and the
#                                             metrics (the paper protocol by default: per track, query frame and earlier
#                                             frames not evaluated, static/dynamic mean on kubric / dex_ycb; add
#                                             --legacy-metrics there for the previous pooled definition)
#   2. scripts/merge_eval_summaries.py        one summary.json / metrics.md over all datasets
# For every network in NETWORKS (default: the main fine-tuned network only).
#
# Usage:
#   bash tracktail_cached.sh [DATASETS]        default: kubric-multiview dex_ycb cmupanoptic_3dgs
#
# Examples:
#   bash tracktail_cached.sh
#   bash tracktail_cached.sh "dex_ycb cmupanoptic_3dgs"
#   NETWORKS="main_pretrain_e25w9v86 ablation_latent_dim_256_lczwsdx3" bash tracktail_cached.sh
#   NETWORKS="all" bash tracktail_cached.sh        every network listed in RUNS below
#
# Results: $OUT_BASE/<network>_canonical_cached/ (for the default network: main_finetune_vzkjqrys_canonical_cached).
# Env overrides: NETWORKS (names from RUNS below, separated by spaces; "all" for every one), OUT_BASE, POSE_FILE (default
# pose3d_cached.npz), DEVICE (default cuda:0).

set -e

DATASETS=${1:-"kubric-multiview dex_ycb cmupanoptic_3dgs"}
POSE_FILE=${POSE_FILE:-pose3d_cached.npz}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f ~/.setup_env_vars.sh ]] && source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

WANDB_BASE=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb
WANDB_BASE_TEST=/groups/karashchuk/home/karashchukl/results/posetail-pretraining-v3/wandb
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs}

# network name (as in format_tables.ipynb) -> its wandb run folder. Pick the ones to run with NETWORKS
declare -A RUNS=(
    [main_finetune_vzkjqrys]="$WANDB_BASE/run-20260912_115732-vzkjqrys"
    [main_pretrain_e25w9v86]="$WANDB_BASE/run-20260908_175758-e25w9v86"
    [ablation_dataset_static_kub_only_f4wpvece]="$WANDB_BASE/run-20260908_175634-f4wpvece"
    [ablation_latent_dim_256_lczwsdx3]="$WANDB_BASE/run-20260908_175759-lczwsdx3"
    [ablation_latent_dim_1024_wk2i4o1u]="$WANDB_BASE/run-20260908_175754-wk2i4o1u"
    [ablation_cam_self_attn_folhop3e]="$WANDB_BASE/run-20260908_175833-folhop3e"
    [ablation_encoder_size_base_hv8nguwx]="$WANDB_BASE/run-20260908_175754-hv8nguwx"
    [ablation_encoder_size_giant_ubk4lvez]="$WANDB_BASE/run-20260908_175759-ubk4lvez"
    [ablation_encoder_size_gigantic_u1c34ou7]="$WANDB_BASE/run-20260908_175807-u1c34ou7"
    # [test_pretrain_gucz6yit]="$WANDB_BASE_TEST/run-20260904_204357-gucz6yit"
)
NETWORKS=${NETWORKS:-main_finetune_vzkjqrys}
if [[ "$NETWORKS" == "all" ]]; then NETWORKS="${!RUNS[*]}"; fi
for NAME in $NETWORKS; do
    if [[ -z "${RUNS[$NAME]}" ]]; then echo "unknown network '$NAME'; known: ${!RUNS[*]}"; exit 1; fi
done

declare -A CANONICAL_CAMS=(
    [kubric-multiview]="view_0 view_1 view_2 view_3"
    [dex_ycb]="view_00 view_01 view_02 view_03"
    [cmupanoptic_3dgs]="1 7 14 20"
)

cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

for NAME in $NETWORKS; do
    OUT="$OUT_BASE/${NAME}_canonical_cached"
    echo "starting tracktail eval on cached tracks ($POSE_FILE): $NAME, $DATASETS -> $OUT"

    for DATASET in $DATASETS; do
        $PYTHON scripts/eval_testset_metrics.py \
            --wandb-folder "${RUNS[$NAME]}" \
            --out           "$OUT" \
            --datasets      "$DATASET" \
            --view-seed     0 \
            --device        "${DEVICE:-cuda:0}" \
            --pose-file     "$POSE_FILE" \
            --cam-names     ${CANONICAL_CAMS[$DATASET]}
    done

    # every call above rewrites summary.json with its own dataset only: rebuild the combined one
    $PYTHON scripts/merge_eval_summaries.py "$OUT"
done

echo "done!"
