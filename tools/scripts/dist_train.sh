#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
PORT=${PORT:-29500}
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=$PORT  train.py  --launcher pytorch ${PY_ARGS} --epochs 30 --ckpt_save_interval 1

# python train.py \
#     --cfg_file cfgs/radar_distill/radar_distill_train.yaml \
#     --launcher none \
#     --epochs 0 \
#     --ckpt_save_interval 1 \
#     --pretrained_model ../ckpt/pillarnet_fullset_init.pth