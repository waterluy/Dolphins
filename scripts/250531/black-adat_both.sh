#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

output="wlu_outputs/${folder_name}"

json_file="test.json"
# 使用jq解析JSON文件
key_name=${script_name#*-}
FORWARD_TYPE=$(jq -r ".${key_name}[0]" "$json_file")
ckpt=$(jq -r ".${key_name}[1]" "$json_file")

# 定义要遍历的methods列表
# methods=("advclip" "anyattack" "sga" "vlpattack" "attackvlm") 
methods=("anyattack")

# 遍历每个method
for method in "${methods[@]}"; do
    echo "Processing method: $method"

    exp_name="${output}/${method}-${key_name}"
    mkdir -p $exp_name

    python vs_attack/dolphins_bench_attack_general.py \
    --method $method \
    --output $exp_name \
    --ckpt $ckpt \
    --forward_type $FORWARD_TYPE

    python tools/dolphin_evaluate.py \
    --exp ${exp_name}/dolphin_output.json \
    --api 'aihub' \
    --gpt 'gpt-4o'

    echo "Completed processing for method: $method"
    echo "----------------------------------------"
done

echo "All methods processed."

