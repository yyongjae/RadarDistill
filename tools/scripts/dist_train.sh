#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
PORT=${PORT:-29500}
# export CUDA_LAUNCH_BLOCKING=1

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=$PORT  train.py  --launcher pytorch ${PY_ARGS}

