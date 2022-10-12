#!/usr/bin/env bash
set -euo pipefail
export ROCR_VISIBLE_DEVICES=1
# use this instead of --resume when starting from just a checkpoint
# --actual_resume path_to_sd_v_1_4_with_ema.ckpt -n 0 \
# if you have an existing log directory, --resume with that directory
# picks the newest checkpoint and will keep global steps etc



exec python lun/main.py \
  model.base_learning_rate=2.5e-6 lightning.trainer.accumulate_grad_batches=1 data.params.batch_size=2 \
 --base ./lun/configs/training-lightning.yaml ./lun/configs/training-txt2img.yaml ./lun/local/training-derpi-sfw.yaml \
 --train=True \
 --accelerator gpu --devices 0, \
 --resume logs/pony-sfw-finetune/2022-09-26T21-18-20_v1.4/ \
 --logdir logs/pony-sfw-finetune/ \
 --data_root /inputs/derpibooru/ --datadir_in_name=False \
 "$@"
