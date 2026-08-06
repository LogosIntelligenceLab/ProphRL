#!/usr/bin/env bash

set -euo pipefail

WORLD_MODEL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

require_dir() {
  local name="$1"
  require_env "${name}"
  if [[ ! -d "${!name}" ]]; then
    echo "${name} is not a directory: ${!name}" >&2
    exit 2
  fi
}

require_checkpoint_run() {
  local name="$1"
  require_dir "${name}"
  local latest="${!name}/checkpoints/latest_checkpoint.txt"
  if [[ ! -f "${latest}" ]]; then
    echo "Missing checkpoint pointer: ${latest}" >&2
    exit 2
  fi
  local checkpoint_file
  checkpoint_file=$(<"${latest}")
  if [[ ! -f "${!name}/checkpoints/model/${checkpoint_file}" ]]; then
    echo "Missing model checkpoint: ${!name}/checkpoints/model/${checkpoint_file}" >&2
    exit 2
  fi
}

launch_training() {
  local config="$1"
  local experiment="$2"
  shift 2

  local gpus="${GPUS:-8}"
  local nnodes="${NNODES:-1}"
  local node_rank="${NODE_RANK:-0}"
  local master_addr="${MASTER_ADDR:-127.0.0.1}"
  local master_port="${MASTER_PORT:-12341}"

  cd "${WORLD_MODEL_ROOT}"
  torchrun \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    --nproc_per_node="${gpus}" \
    -m scripts.train \
    --config="${config}" \
    -- experiment="${experiment}" "$@"
}
