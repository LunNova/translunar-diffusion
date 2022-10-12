#!/bin/bash
set -euo pipefail
export ROCR_VISIBLE_DEVICES=1

# use this instead of --resume for first start
# --actual_resume models/sd-v1.4-kl.ckpt -n 0 \

exec python lun/main.py \
  model.base_learning_rate=5e-6 \
 --base ./lun/configs/training-lightning.yaml ./lun/configs/training-encoder.yaml \
 --train \
 --accelerator gpu --devices 0, \
 --resume logs/encoder-kl-f8-derpibooru/2022-09-27T11-39-13_0/ \
 --logdir logs/encoder-kl-f8-derpibooru/ \
 --data_root /inputs/derpibooru/ --datadir_in_name=False
