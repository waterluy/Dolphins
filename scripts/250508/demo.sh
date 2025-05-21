#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

output="wlu_outputs/${folder_name}"
mkdir -p $output

# 定义可变参数范围
ckpt="ckpts/20250509/sat/new_checkpoint/checkpoint2.pt"
exp_name="${output}/${script_name}"

python inference.py \
 --ckpt "$ckpt" \
 --forward_type 0

# python tools/dolphin_evaluate.py \
#  --exp ${exp_name}/dolphin_output.json \
#  --api 'aihub' \
#  --gpt 'gpt-4o'







