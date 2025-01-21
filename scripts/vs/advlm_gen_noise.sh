#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

# 定义可变参数范围
eps=0.1
steps=50
method=4

# 定义固定参数
dire="pos"
lp="linf"
samples=3
output="exr-main"
affine=True

python exr/dolphins_bench_attack_exr_gen_noise.py --samples $samples --method $method --affine  --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1



