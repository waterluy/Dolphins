#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

output=results1117
lamb1=1.0
lamb2=1.0
lamb3=0.0
eps=0.13

python attack/wlu-judge-offline-mloss-i2.py --output $output --sup-text --sup-clean  --eps $eps --iter 20 --query 8 --loss cos --lamb1 $lamb1 --lamb2 $lamb2 --lamb3 $lamb3
python tools/dolphin_evaluate.py --exp ${output}/bench_attack_coi-judge-offline-cos-i2-text-clean_eps${eps}_iter20_query8_lamb1-${lamb1}_lamb2-${lamb2}_lamb3-${lamb3}/dolphin_output.json


