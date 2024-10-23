#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# 定义可变参数范围
eps_values=(0.2)
iter_values=(10 100)
transfer_values=('drivelm')

# 运行实验
for transfer in "${transfer_values[@]}"; do
    for eps in "${eps_values[@]}"; do
        # fgsm
        # 定义输出文件路径
        fgsm_output_file="results/bench_attack_black_${transfer}_fgsm_eps${eps}/dolphin_output.json"
        
        # 检查FGSM实验的输出文件是否存在
        if [ ! -f "$fgsm_output_file" ]; then
            echo "Running experiment FGSM with transfer=$transfer eps=$eps"
            python dolphins_bench_attack_black.py --method fgsm --transfer "$transfer" --eps "$eps"    # >> output_eps_${eps}_iter_${iter}.log 2>&1
            python tools/dolphin_evaluate.py --exp bench_attack_black_${transfer}_fgsm_eps${eps}
            echo "Finished experiment FGSM with transfer=$transfer eps=$eps, steps=$steps"
        else
            echo "Skipping FGSM experiment with transfer=$transfer eps=$eps, steps=$steps as output already exists"
        fi

        for iter in "${iter_values[@]}"; do
            # pgdlinf
            # 定义输出文件路径
            pgd_output_file="results/bench_attack_black_${transfer}_pgdlinf_eps${eps}_steps${steps}/dolphin_output.json"
            
            # 检查PGD实验的输出文件是否存在
            if [ ! -f "$pgd_output_file" ]; then
                echo "Running experiment PGD with transfer=$transfer eps=$eps, steps=$steps"
                python dolphins_bench_attack_black.py --method pgdlinf --transfer "$transfer" --eps "$eps" --steps "$steps"     # >> output_eps_${eps}_iter_${iter}.log 2>&1
                python tools/dolphin_evaluate.py --exp bench_attack_black_${transfer}_pgdlinf_eps${eps}_steps${steps}
                echo "Finished experiment PGD with transfer=$transfer eps=$eps, steps=$steps"
            else
                echo "Skipping PGD experiment with transfer=$transfer eps=$eps, steps=$steps as output already exists"
            fi
        done
    done
done

# python tools/fill_table.py

