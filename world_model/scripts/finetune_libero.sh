#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

require_dir PROPHRL_LIBERO_DATA
require_checkpoint_run PROPHRL_PRETRAIN_CKPT

launch_training \
  cosmos_predict2/configs/base/config_libero.py \
  prophet_finetune_libero \
  "$@"
