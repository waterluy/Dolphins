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

lamb1=0.75
lamb2=0.75
lamb3=0.05
eps=0.13
method=adapter_tuning
epoch=0
# ckpt="ckpts/20250508/adapter_tuning/new_checkpoint/checkpoint1.pt"
ckpt="ckpts/20250509/adapterRes_tuning/llava_bddx/step_540/checkpoint540.pt"
exp_name="${output}/${method}${epoch}-${script_name}"

python attack/final.py \
 --output $exp_name \
 --sup-text \
 --sup-clean \
 --sup-adj \
 --eps $eps \
 --iter 40 \
 --query 1 \
 --loss cos \
 --lamb1 $lamb1 \
 --lamb2 $lamb2 \
 --lamb3 $lamb3 \
 --ckpt $ckpt \
 --forward_type 7

python tools/dolphin_evaluate.py \
 --exp ${exp_name}/dolphin_output.json \
 --api 'aihub' \
 --gpt 'gpt-4o'


