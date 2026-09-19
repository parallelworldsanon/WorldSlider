#!/usr/bin/env bash
# The three training stages, run from a cosmos-transfer2.5 checkout that has
# flightgrid_control.patch applied and worldslider/cosmos/*.py installed under
# cosmos_transfer2/experiments/worldslider/ (see code/README.md).
set -euo pipefail
export WORLDSLIDER_DATASET=${WORLDSLIDER_DATASET:-datasets/worldslider}
export PYTHONPATH=${PYTHONPATH:-}:$(dirname "$0")/..     # worldslider.reward for stage 3
N=${NGPUS:-4}

case "${1:-}" in
  adapt)   # stage 1: control adaptation from the official depth-control checkpoint
    torchrun --standalone --nproc_per_node=$N -m scripts.train \
      --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
      -- experiment=worldslider_control_adaptation ;;
  distill) # stage 2: 4-step DMD distillation (WORLDSLIDER_TEACHER = consolidated stage-1 .pt)
    torchrun --standalone --nproc_per_node=$N -m scripts.train \
      --config=cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py \
      -- experiment=worldslider_dmd ;;
  align)   # stage 3: reward-weighted DMD (WORLDSLIDER_STUDENT = stage-2 checkpoint)
    torchrun --standalone --nproc_per_node=$N -m scripts.train \
      --config=cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py \
      -- experiment=worldslider_flight_alignment ;;
  *) echo "usage: $0 {adapt|distill|align}"; exit 1 ;;
esac
