#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

[ -f cluster/setup_env_vars.sh ] && . cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


# bsub -J en6-$d -e ~/logs/posetail/en6-$d.err -o ~/logs/posetail/en6-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash cluster/train_script.sh configs/archive/config_encoder_pretrain.toml

# Paper pretrain (configs/paper/pretrain.toml; paper run used 4x H100).
bsub -J pretrain-$d -e ~/logs/posetail/pretrain-$d.err -o ~/logs/posetail/pretrain-$d.out \
    -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/paper/pretrain.toml
