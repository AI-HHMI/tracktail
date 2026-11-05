# Shared setup for the paper drivers (sourced, not executed):
#   source "$REPO/scripts/paper_common.sh"
# Sources paper.env, picks the pixi Python, and maps paper network names to model arguments for
# scripts/eval_testset_metrics.py (Hugging Face by default; TRACKTAIL_WANDB_BASE switches to W&B run dirs).

PAPER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../paper.env
source "$PAPER_REPO/paper.env"
export LD_LIBRARY_PATH="$PAPER_REPO/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
if [[ -x "$PAPER_REPO/.pixi/envs/default/bin/python" ]]; then
    PIXI_PYTHON="$PAPER_REPO/.pixi/envs/default/bin/python"
else
    PIXI_PYTHON="pixi run --manifest-path $PAPER_REPO/pyproject.toml python"
fi

# network name (= output folder prefix used by tracktail-paper) -> HF subfolder of $TRACKTAIL_PAPER_CKPTS_REPO,
# and the W&B run dir of the original training run (fallback when TRACKTAIL_WANDB_BASE is set).
# main_finetune_vzkjqrys is the paper model; it loads from $TRACKTAIL_HF_REPO@$TRACKTAIL_HF_REVISION
# (identical to $TRACKTAIL_PAPER_CKPTS_REPO/finetune).
declare -A PAPER_HF_SUBFOLDER=(
    [main_finetune_vzkjqrys]=""
    [main_pretrain_e25w9v86]="pretrain"
    [ablation_cam_self_attn_folhop3e]="ablation-cam-attn-off"
    [ablation_temporal_self_attn_b3de55ap]="ablation-temporal-attn-off"
    [ablation_cam_temporal_self_attn_cnmw9dcf]="ablation-cam-temporal-attn-off"
    [ablation_encoder_size_base_hv8nguwx]="ablation-encoder-base"
    [ablation_encoder_size_giant_ubk4lvez]="ablation-encoder-giant"
    [ablation_encoder_size_gigantic_u1c34ou7]="ablation-encoder-gigantic"
    [ablation_latent_dim_256_lczwsdx3]="ablation-latent-256"
    [ablation_latent_dim_1024_wk2i4o1u]="ablation-latent-1024"
)
declare -A PAPER_WANDB_RUN=(
    [main_finetune_vzkjqrys]="run-20260912_115732-vzkjqrys"
    [main_pretrain_e25w9v86]="run-20260908_175758-e25w9v86"
    [ablation_cam_self_attn_folhop3e]="run-20260908_175833-folhop3e"
    [ablation_temporal_self_attn_b3de55ap]="run-20261103_121045-b3de55ap"
    [ablation_cam_temporal_self_attn_cnmw9dcf]="run-20261103_121014-cnmw9dcf"
    [ablation_encoder_size_base_hv8nguwx]="run-20260908_175754-hv8nguwx"
    [ablation_encoder_size_giant_ubk4lvez]="run-20260908_175759-ubk4lvez"
    [ablation_encoder_size_gigantic_u1c34ou7]="run-20260908_175807-u1c34ou7"
    [ablation_latent_dim_256_lczwsdx3]="run-20260908_175759-lczwsdx3"
    [ablation_latent_dim_1024_wk2i4o1u]="run-20260908_175754-wk2i4o1u"
)
PAPER_NETWORKS="main_finetune_vzkjqrys main_pretrain_e25w9v86 ablation_cam_self_attn_folhop3e ablation_temporal_self_attn_b3de55ap ablation_cam_temporal_self_attn_cnmw9dcf ablation_encoder_size_base_hv8nguwx ablation_encoder_size_giant_ubk4lvez ablation_encoder_size_gigantic_u1c34ou7 ablation_latent_dim_256_lczwsdx3 ablation_latent_dim_1024_wk2i4o1u"

# model_args NAME -> prints the model arguments for eval_testset_metrics.py / tracktail_tapvid3d_official.py
model_args() {
    local name=$1
    if [[ -z "${PAPER_WANDB_RUN[$name]+x}" ]]; then
        echo "unknown network '$name'; known: $PAPER_NETWORKS" >&2; return 1
    fi
    if [[ -n "${TRACKTAIL_WANDB_BASE:-}" ]]; then
        echo "--wandb-folder $TRACKTAIL_WANDB_BASE/${PAPER_WANDB_RUN[$name]}"
    elif [[ -z "${PAPER_HF_SUBFOLDER[$name]}" ]]; then
        echo "--hf-repo $TRACKTAIL_HF_REPO --revision $TRACKTAIL_HF_REVISION"
    else
        echo "--hf-repo $TRACKTAIL_PAPER_CKPTS_REPO --revision $TRACKTAIL_PAPER_CKPTS_REVISION --subfolder ${PAPER_HF_SUBFOLDER[$name]}"
    fi
}

# Results trees (same subfolder names as the paper's results; the Tracktail tree was called
# posetail-outputs-vis0.1-occunknown on disk)
TRACKTAIL_OUT_BASE="${TRACKTAIL_OUT_BASE:-$TRACKTAIL_RESULTS/tracktail-outputs}"
MVTRACKER_OUT_BASE="${MVTRACKER_OUT_BASE:-$TRACKTAIL_RESULTS/mvtracker-outputs}"
COTRACKER_OUT_BASE="${COTRACKER_OUT_BASE:-$TRACKTAIL_RESULTS/cotracker-outputs}"
ARROW_OUT_BASE="${ARROW_OUT_BASE:-$TRACKTAIL_RESULTS/arrow-outputs}"
