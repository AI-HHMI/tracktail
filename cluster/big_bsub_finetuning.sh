#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

[ -f cluster/setup_env_vars.sh ] && . cluster/setup_env_vars.sh

d=$(date +d%dh%H)

# Paper finetune (configs/paper/finetune.toml; paper run used 8x H100; needs PRETRAIN_CKPT = your pretrain checkpoint).
bsub -J finetune-$d -e ~/logs/tracktail/finetune-$d.err -o ~/logs/tracktail/finetune-$d.out \
    -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 168:00 \
    /bin/bash cluster/train_script.sh configs/paper/finetune.toml
