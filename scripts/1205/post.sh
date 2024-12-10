#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本的文件名（不包括扩展名）
script_name=$(basename "$0" .sh)

output="results${folder_name}"
lamb1=0.75
lamb2=0.75
lamb3=0.05
eps=0.1


python defense/defense_base.py --defense $script_name --output $output  --sup-text --sup-clean --sup-adj --eps $eps --iter 20 --query 8 --loss cos --lamb1 $lamb1 --lamb2 $lamb2 --lamb3 $lamb3
python tools/dolphin_evaluate.py --api aihub --gpt gpt-4o --exp ${output}/defense_${script_name}/dolphin_output.json


