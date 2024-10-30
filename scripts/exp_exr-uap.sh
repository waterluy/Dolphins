#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 定义可变参数范围
eps_values=(0.1 )
steps_values=(5 10 )
# steps_values=(1 2 4 10)
method_values=(2 3 4)

# 定义固定参数
dire="pos"
lp="linf"

# 运行实验
for eps in "${eps_values[@]}"; do
    for steps in "${steps_values[@]}"; do
        for method in "${method_values[@]}"; do
            # 定义输出文件路径
            exr_output_file="results/bench_attack_m${method}-uap_white_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
            exr_score_file="results/bench_attack_m${method}-uap_white_${lp}_eps${eps}_steps${steps}_${dire}/bench_score.csv"
            
            # 检查EXR实验的输出文件是否存在
            if [ ! -f "$exr_score_file" ]; then
                echo "Running experiment m${method}-uap with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
                python exr/dolphins_bench_attack_exr-uap.py --eps "$eps" --steps "$steps" --method $method --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
            fi
            if [ ! -f "$exr_score_file" ]; then
                python tools/dolphin_evaluate.py --exp bench_attack_m${method}-uap_white_${lp}_eps${eps}_steps${steps}_${dire}
                echo "Finished experiment m${method}-uap with eps=$eps, steps=$steps"
            else
                echo "Skipping m${method}-uap experiment with eps=$eps, steps=$steps as output already exists"
            fi
        done
        pgd_output_file="results/bench_attack_pgd-uap_white_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
        pgd_score_file="results/bench_attack_pgd-uap_white_${lp}_eps${eps}_steps${steps}_${dire}/bench_score.csv"
        # 检查PGD实验的输出文件是否存在
        if [ ! -f "$pgd_output_file" ]; then
            echo "Running experiment PGD-uap with eps=$eps, steps=$steps dire=$dire, lp=$lp"
            python exr/dolphins_bench_attack_pgd_white-uap.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        fi
        if [ ! -f "$pgd_score_file" ]; then 
            python tools/dolphin_evaluate.py --exp bench_attack_pgd-uap_white_${lp}_eps${eps}_steps${steps}_${dire}
            echo "Finished experiment PGD-uap with eps=$eps, steps=$steps"
        else
            echo "Skipping PGD-uap experiment with eps=$eps, steps=$steps as output already exists"
        fi
    done
done

python tools/fill_table.py

