#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# 定义可变参数范围
eps_values=(0.2 0.1 0.5 0.05)
iter_values=(20 50 100)
query=(8 10 5)

# 运行实验
for eps in "${eps_values[@]}"; do
    for iter in "${iter_values[@]}"; do
        # 定义输出文件路径
        coi_output_file="results/bench_attack_coi_eps${eps}_iter${iter}_query${query}/dolphin_output.json"
        pgd_output_file="results/bench_attack_coi_eps${eps}_iter${iter}_query${query}/dolphin_output.json"
        
        # 检查COI实验的输出文件是否存在
        if [ ! -f "$coi_output_file" ]; then
            echo "Running experiment COI with eps=$eps, iter=$iter query=$query"
            python dolphins_bench_attack_wlu.py --eps "$eps" --iter "$iter" --query "$query"    # >> output_eps_${eps}_iter_${iter}.log 2>&1
            python tools/dolphin_evaluate.py --exp bench_attack_coi_eps${eps}_iter${iter}_query${query}
            echo "Finished experiment COI with eps=$eps, iter=$iter"
        else
            echo "Skipping COI experiment with eps=$eps, iter=$iter as output already exists"
        fi


    done
done

# python tools/fill_table.py

