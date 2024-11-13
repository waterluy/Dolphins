#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
eps=0.1
steps=50
method=4
dire="pos"
lp="linf"
samples=3
affine=True
output="exr-uni"
# python exr/dolphins_bench_attack_fgsm_white-uni.py --output $output --eps "$eps" --dire "$dire"  # >> output_eps_${eps}_steps_${steps}.log 2>&1
python tools/dolphin_evaluate.py --gpt gpt-3.5-turbo-0613  --exp ${output}/bench_attack_fgsmuni_white_eps${eps}_${dire}/dolphin_output.json