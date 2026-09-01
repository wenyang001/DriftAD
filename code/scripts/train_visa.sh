#!/bin/bash
# Train DriftAD on VisA.
set -e
cd "$(dirname "$0")/.."

export VISA_ROOT=${VISA_ROOT:-../data/visa}

deepspeed --master_port 60336 train_visa.py \
    --model openllama_peft \
    --stage 1 \
    --epochs 80 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --image_root_path ../data/images/ \
    --save_path ./ckpt/train_visa/ \
    --log_path ./ckpt/train_visa/log_rest/
