#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

[ -f cluster/setup_env_vars.sh ] && . cluster/setup_env_vars.sh

module load cuda/12.8

# Paper configs read TRACKTAIL_DATA / TRACKTAIL_RESULTS (paper.env) and PRETRAIN_CKPT (finetune).
[ -f paper.env ] && . ./paper.env

pixi run python train.py --config-path $1 \
    --devices -1 --strategy ddp_find_unused_parameters_true

