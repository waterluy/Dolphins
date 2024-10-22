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
        echo "Running experiment EXR with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
        python dolphins_bench_attack_exr.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        python tools/dolphin_evaluate.py --exp bench_attack_exr_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment EXR with eps=$eps, steps=$steps"

        echo "Running experiment PGD with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
        python dolphins_bench_attack_pgd_white.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        python tools/dolphin_evaluate.py --exp bench_attack_pgd_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment PGD with eps=$eps, steps=$steps"
    done
done

python tools/fill_table.py

