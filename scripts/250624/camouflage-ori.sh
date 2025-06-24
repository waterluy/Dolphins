#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

output="wlu_outputs/${folder_name}"
mkdir -p $output

# 定义可变参数范围
exp_name="${output}/${script_name}"

json_file="test.json"
# 使用jq解析JSON文件
key_name=${script_name#*-}
FORWARD_TYPE=$(jq -r ".${key_name}[0]" "$json_file")
# CKPT=$(jq -r ".${key_name}[1]" "$json_file")

# python vs_attack/camouflage.py \
#  --output $exp_name \
#  --forward_type $FORWARD_TYPE \
#  --image-folder "playground/camouflage/test_our"

python tools/camouflage_evaluate.py \
 --exp ${exp_name}/dolphin_output.json \
 --api 'aihub' \
 --gpt 'gpt-3.5-turbo'