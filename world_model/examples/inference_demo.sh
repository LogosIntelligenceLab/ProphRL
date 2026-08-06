#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <libero|bridge|pretrained> <input-video> <actions.json|actions.npz> <base-checkpoints> [output.mp4]" >&2
  exit 2
fi

variant=$1
input_video=$2
actions=$3
base_checkpoints=$4
output=${5:-outputs/${variant}_demo.mp4}

prophet-demo \
  --variant "${variant}" \
  --input "${input_video}" \
  --actions "${actions}" \
  --base-checkpoints "${base_checkpoints}" \
  --output "${output}"
