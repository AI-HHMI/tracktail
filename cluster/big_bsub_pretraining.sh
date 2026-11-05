#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

[ -f cluster/setup_env_vars.sh ] && . cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)

# Paper pretrain (configs/paper/pretrain.toml; paper run used 4x H100).
bsub -J pretrain-$d -e ~/logs/tracktail/pretrain-$d.err -o ~/logs/tracktail/pretrain-$d.out \
    -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/paper/pretrain.toml
