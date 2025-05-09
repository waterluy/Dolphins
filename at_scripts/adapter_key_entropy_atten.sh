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

FORWARD_TYPE=$(jq -r ".${script_name}" "$JSON_FILE")
# 如果读取结果为空或为 null，则报错并退出
if [ -z "$FORWARD_TYPE" ] || [ "$FORWARD_TYPE" == "null" ]; then
  echo "❌ Error: 无法在 $JSON_FILE 中找到 key \"$script_name\" 对应的 FORWARD_TYPE"
  exit 1
fi

accelerate launch --multi_gpu \
 --num_processes $GPU_NUM \
pipeline/at_train.py \
 --use_lora \
 --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
 --at_iter 10 \
 --at_eps_imgs 0.1 \
 --forward_type $FORWARD_TYPE

 