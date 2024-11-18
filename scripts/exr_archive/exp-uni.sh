#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
eps=0.1
steps=50
method=4
dire="pos"
lp="linf"
samples=3
affine=True
output="exr-uni"
# python exr/dolphins_bench_attack_exr-uni.py --samples $samples --method $method --affine --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp"
python tools/dolphin_evaluate.py --gpt gpt-3.5-turbo-0613  --exp $output/bench_attack_m${method}uni-${affine}_white_${lp}_eps${eps}_steps${steps}_samples${samples}_${dire}/dolphin_output.json