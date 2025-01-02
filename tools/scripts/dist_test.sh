#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
PORT=${PORT:-29500}


python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=$PORT test.py --launcher pytorch ${PY_ARGS}

