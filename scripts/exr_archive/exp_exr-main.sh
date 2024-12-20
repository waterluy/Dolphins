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

# 运行实验
inf_output_file="${output}/bench_inference/dolphin_output.json"

# 检查inf实验的输出文件是否存在
if [ ! -f "$inf_output_file" ]; then
    echo "dolphins_bench_inference.py --output $output"
    python dolphins_bench_inference.py --output $output
    python tools/dolphin_evaluate.py --exp $inf_output_file
else
    echo "Skipping inf experiment as output already exists"
fi

# 定义输出文件路径
exr_output_file="${output}/bench_attack_m${method}-${affine}_white_${lp}_eps${eps}_steps${steps}_samples${samples}_${dire}/dolphin_output.json"

# 检查EXR实验的输出文件是否存在
if [ ! -f "$exr_output_file" ]; then
    echo "python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --affine --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp""
    python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --affine --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
    python tools/dolphin_evaluate.py --exp $exr_output_file
else
    echo "Skipping EXR experiment with eps=$eps, steps=$steps as output already exists"
fi

affine=False
method=2
samples=1
pgd_output_file="${output}/bench_attack_m${method}-${affine}_white_${lp}_eps${eps}_steps${steps}_samples${samples}_${dire}/dolphin_output.json"

# 检查PGD-m2实验的输出文件是否存在
if [ ! -f "$pgd_output_file" ]; then
    echo "python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp""
    python exr/dolphins_bench_attack_exr.py --samples $samples --method $method --output $output --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
    python tools/dolphin_evaluate.py --exp $pgd_output_file
else
    echo "Skipping PGD experiment with eps=$eps, steps=$steps as output already exists"
fi

fgsm_output_file="${output}/bench_attack_fgsm_white_eps${eps}_${dire}/dolphin_output.json"
# 检查FGSM实验的输出文件是否存在
if [ ! -f "$pgd_output_file" ]; then
    echo "Running experiment FGSM with eps=$eps dire=$dire, lp=$lp"
    python exr/dolphins_bench_attack_fgsm_white.py --output $output --eps "$eps" --dire "$dire"  # >> output_eps_${eps}_steps_${steps}.log 2>&1
    python tools/dolphin_evaluate.py --exp $fgsm_output_file
else
    echo "Skipping FGSM experiment with eps=$eps, steps=$steps as output already exists"
fi





