#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

[ -f cluster/setup_env_vars.sh ] && . cluster/setup_env_vars.sh

# LSF launcher for a finetuning run (6 GPUs on one host); runs cluster/train_script.sh.
# Usage: sh cluster/big_bsub_finetuning.sh [config.toml]   (default: the tracktail-static-animal finetune)
CONFIG=${1:-configs/config_tracktail_finetuning_v6.toml}

d=$(date +d%dh%H)

bsub -J finetune-$d -e ~/logs/tracktail/finetune-$d.err -o ~/logs/tracktail/finetune-$d.out \
    -n 72 -q gpu_h100 -R "span[hosts=1]" -gpu "num=6" -W 96:00 \
    /bin/bash cluster/train_script.sh "$CONFIG"
