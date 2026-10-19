#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then source source/env.sh; fi
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=29600
DATA_DIR="${DATA_DIR:-}"
if [[ -z "$DATA_DIR" ]]; then
  echo "DATA_DIR is required: set it to a CIFAR-10 directory containing cifar-10-batches-py" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]]; then
  echo "Missing CIFAR-10: $DATA_DIR/cifar-10-batches-py (set DATA_DIR)" >&2
  exit 2
fi
RUN_NAME="${RUN_NAME:-vit_$(basename "$PWD")_8xnpu_$(date +%Y%m%d_%H%M%S)_$$}"
exec torchrun --standalone --nproc_per_node=8 source/train_perf.py \
  --config source/training_config.yaml --data-dir "$DATA_DIR" \
  --output "${OUTPUT_DIR:-$PWD/archive/runs}" --experiment "$RUN_NAME" "$@"
