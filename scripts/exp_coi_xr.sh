#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 定义可变参数范围
eps_values=(0.2)
iter_values=(20)
query_values=(8)

# 运行实验
for eps in "${eps_values[@]}"; do
    for iter in "${iter_values[@]}"; do
        for query in "${query_values[@]}"; do
            # without stage1(chain2)
            # 定义输出文件路径
            # result=$((iter * query))
            # coi_output_file="results/bench_attack_coi-wo-stage1_eps${eps}_iter${result}/dolphin_output.json"
                    
            # # 检查COI实验的输出文件是否存在
            # if [ ! -f "$coi_output_file" ]; then
            #     echo "Running experiment COI with eps=$eps, iter=$iter query=$query result=$result"
            #     python attack/dolphins_bench_attack_wlu_xr_wo_stage1.py --eps "$eps" --iter "$iter" --query "$query"    # >> output_eps_${eps}_iter_${iter}.log 2>&1
            #     python tools/dolphin_evaluate.py --exp bench_attack_coi-wo-stage1_eps${eps}_iter${result}
            #     echo "Finished experiment COI with eps=$eps, iter=$iter"
            # else
            #     echo "Skipping COI experiment with eps=$eps, iter=$iter as output already exists"
            # fi

            # without stage2(chain1)
            # 定义输出文件路径
            result=$((iter * query))
            coi_output_file="results/bench_attack_coi-wo-stage2_eps${eps}_iter${result}/dolphin_output.json"
                    
            # 检查COI实验的输出文件是否存在
            if [ ! -f "$coi_output_file" ]; then
                echo "Running experiment COI with eps=$eps, iter=$iter query=$query result=$result"
                python attack/dolphins_bench_attack_wlu_xr_wo_stage2.py --eps "$eps" --iter "$iter" --query "$query"    # >> output_eps_${eps}_iter_${iter}.log 2>&1
                python tools/dolphin_evaluate.py --exp bench_attack_coi-wo-stage2_eps${eps}_iter${result}
                echo "Finished experiment COI with eps=$eps, iter=$iter"
            else
                echo "Skipping COI experiment with eps=$eps, iter=$iter as output already exists"
            fi

        done
    done
done

# python tools/fill_table.py

