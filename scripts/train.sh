#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SAM3_SOURCE=${SAM3_SOURCE:-$PROJECT_ROOT/vendor/sam3}
export PYTHONPATH="$PROJECT_ROOT:$SAM3_SOURCE:${PYTHONPATH:-}"
export SAM3_MAX_EPOCHS=${SAM3_MAX_EPOCHS:-4}
export SAM3_LOCAL_BATCH=${SAM3_LOCAL_BATCH:-1}
export SAM3_GRAD_ACCUM_STEPS=${SAM3_GRAD_ACCUM_STEPS:-1}
export SAM3_FREEZE_POLICY=${SAM3_FREEZE_POLICY:-tfdm}
export SAM3_SCHEDULER_TIMESCALE=${SAM3_SCHEDULER_TIMESCALE:-200}
export SAM3_LR_TRANSFORMER=${SAM3_LR_TRANSFORMER:-1.6e-4}
export SAM3_LR_TEXT=${SAM3_LR_TEXT:-1e-5}
export SAM3_LR_VISION=${SAM3_LR_VISION:-5e-5}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [[ ${NNODES:-1} == 1 ]]; then
    torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-1}" "$PROJECT_ROOT/scripts/train.py"
else
    : "${NODE_RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"
    torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" \
      --nproc_per_node="${GPUS_PER_NODE:-8}" --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" "$PROJECT_ROOT/scripts/train.py"
fi
