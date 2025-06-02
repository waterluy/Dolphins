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

# 定义要遍历的methods列表
methods=("sat" "fastat" "freeat" "trades" 
 "advpt" "sat_key_entropy_atten" 
 "sat_both_key_entropy_atten" 
 "adapterRes_tuning" 
 "adapterRes_both_key_entropy_atten" 
 "adapterRes_key_entropy_atten"
 "adapterRes4visual" 
 "adapterResNoshare_both_key_entropy_atten"
 "adat_both"
 "visualat_both") 

# 遍历每个method
for key_name in "${methods[@]}"; do
    echo "Processing method: $key_name"

    # 使用jq解析JSON文件
    FORWARD_TYPE=$(jq -r ".${key_name}[0]" "$json_file")
    ckpt=$(jq -r ".${key_name}[1]" "$json_file")

    exp_name="${output}/noattack-${key_name}"
    mkdir -p $exp_name

    # python dolphins_bench_inference.py \
    # --output $exp_name \
    # --ckpt $ckpt \
    # --forward_type $FORWARD_TYPE

    python tools/dolphin_evaluate.py \
    --exp ${exp_name}/dolphin_output.json \
    --api 'aihub' \
    --gpt 'gpt-3.5-turbo'

    echo "Completed processing for method: $method"
    echo "----------------------------------------"
done

echo "All methods processed."

