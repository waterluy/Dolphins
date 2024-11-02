#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 定义可变参数范围
eps=0.1
steps=50
method=0

# 定义固定参数
dire="pos"
lp="linf"
samples=3
output="exr-b"
affine=True

# 运行实验
# 定义输出文件路径
exr_output_file="${output}/bench_attack_m${method}-${affine}_white_${lp}_eps${eps}_steps${steps}_samples${samples}_${dire}/dolphin_output.json"

# 检查EXR实验的输出文件是否存在
if [ ! -f "$exr_output_file" ]; then
    echo "python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --affine --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp""
    python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --affine --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
    exp=$(echo "$exr_output_file" | cut -d'/' -f2)
    python tools/dolphin_evaluate.py --exp $exp
else
    echo "Skipping EXR experiment with eps=$eps, steps=$steps as output already exists"
fi
