#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# 定义可变参数范围
eps_values=(0.1 0.2 0.5 0.05)
steps_values=(3 5 10 20 50 100)

# 定义固定参数
dire="pos"
lp="linf"

# 运行实验
for eps in "${eps_values[@]}"; do
    for steps in "${steps_values[@]}"; do
        # 定义输出文件路径
        exr_output_file="results/bench_attack_exr_white_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
        pgd_output_file="results/bench_attack_pgd_white_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
        
        # 检查EXR实验的输出文件是否存在
        if [ ! -f "$exr_output_file" ]; then
            echo "Running experiment EXR with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
            python dolphins_bench_attack_exr.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        else
            echo "Skipping EXR experiment with eps=$eps, steps=$steps as output already exists"
        fi
        python tools/dolphin_evaluate.py --exp bench_attack_exr_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment EXR with eps=$eps, steps=$steps"

        # 检查PGD实验的输出文件是否存在
        if [ ! -f "$pgd_output_file" ]; then
            echo "Running experiment PGD with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
            python dolphins_bench_attack_pgd_white.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        else
            echo "Skipping PGD experiment with eps=$eps, steps=$steps as output already exists"
        fi
        python tools/dolphin_evaluate.py --exp bench_attack_pgd_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment PGD with eps=$eps, steps=$steps"
    done
done

python tools/fill_table.py

