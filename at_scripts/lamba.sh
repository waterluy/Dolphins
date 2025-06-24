#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

GPU_NUM=6
JSON_FILE="forward.json"

# 获取当前日期，格式为YYYYMMDD
current_date=$(date +"%Y%m%d")
echo "当前日期: $current_date"  # 输出示例: 20250416

# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

# VALUES=(0.01 0.05 0.2 0.5 1.0)
VALUES=(0.5 1.0)

for lamb in "${VALUES[@]}"; do
    echo "开始运行 lamb = $lamb"

    OUTPUT_DIR=./ckpts/${current_date}/adapter_both_key_entropy_atten-lamba${lamb}
    mkdir -p $OUTPUT_DIR

    accelerate launch --multi_gpu \
    --num_processes $GPU_NUM \
    pipeline/at_train.py \
    --use_lora \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --at_iter 10 \
    --at_eps_imgs 0.1 \
    --forward_type 15 \
    --lamb $lamb
done
