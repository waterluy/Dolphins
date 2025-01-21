#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

output=results1228
lamb1=0.75
lamb2=0.75
lamb3=0.05
eps=0.1
# python attack/vlmo.py --output results1228  --sup-text --sup-clean --sup-adj --eps 0.1 --iter 160 --query 1 --loss cos --lamb1 1.0 --lamb2 1.0 --lamb3 0.05
python attack/vlmo.py --output $output  --sup-text --sup-clean --sup-adj --eps $eps --iter 10 --query 1 --loss cos --lamb1 $lamb1 --lamb2 $lamb2 --lamb3 $lamb3
# python tools/dolphin_evaluate.py --exp ${output}/vlmo_eps${eps}_iter30_query1/dolphin_output.json


