#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")

output="results${folder_name}"
lamb1=0.75
lamb2=0.75
lamb3=0.05
eps=0.13
iter=20
query=4

python attack/wlu-judge-offline-mloss-i2.py --output $output  --sup-text --sup-clean --sup-adj --eps $eps --iter $iter --query $query --loss cos --lamb1 $lamb1 --lamb2 $lamb2 --lamb3 $lamb3
python tools/dolphin_evaluate.py --exp ${output}/bench_attack_coi-judge-offline-cos-i2-text-clean-adj_eps${eps}_iter${iter}_query${query}_lamb1-${lamb1}_lamb2-${lamb2}_lamb3-${lamb3}/dolphin_output.json


