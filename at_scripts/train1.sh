#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

GPU_NUM=7
JSON_FILE="forward.json"

# 获取当前日期，格式为YYYYMMDD
current_date=$(date +"%Y%m%d")
echo "当前日期: $current_date"  # 输出示例: 20250416

# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

OUTPUT_DIR=./ckpts/${current_date}/${script_name}
mkdir -p $OUTPUT_DIR

accelerate launch --mixed_precision "no" --multi_gpu \
 --num_processes $GPU_NUM \
 pipeline/train1.py \
 --use_lora \
 --output_dir $OUTPUT_DIR \
 --per_device_train_batch_size 2 \
 --num_train_epochs 1
