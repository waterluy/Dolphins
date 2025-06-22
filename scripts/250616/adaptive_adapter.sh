#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本文件的名字
script_name=$(basename "$0" .sh)

output="wlu_outputs/${folder_name}"
mkdir -p $output

# 定义可变参数范围
eps=0.1
steps=10

VALUES=('l1' 'l2')

for mode in "${VALUES[@]}"; do

    exp_name="${output}/${script_name}-${mode}"

    python exr/dolphins_bench_adaptive_adapter.py \
    --output $exp_name \
    --eps "$eps" \
    --steps "$steps" \
    --distance $mode \
    --forward_type 15 \
    --ckpt ckpts/20250522/adapterRes_both_key_entropy_atten/llava_bddx/step_540/checkpoint540.pt 


    python tools/dolphin_evaluate.py \
    --exp ${exp_name}/dolphin_output.json \
    --api 'aihub' \
    --gpt 'gpt-3.5-turbo'
done







