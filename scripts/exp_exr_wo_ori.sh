#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

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
        exrwoori_output_file="results/bench_attack_exrwoori_white_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
        exrwoori1_output_file="results/bench_attack_exrwoori1_${lp}_eps${eps}_steps${steps}_${dire}/dolphin_output.json"
        
        # 检查exrwoori实验的输出文件是否存在
        # if [ ! -f "$exrwoori_output_file" ]; then
        echo "Running experiment exrwoori with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
        python dolphins_bench_attack_exr.py --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        python tools/dolphin_evaluate.py --exp bench_attack_exrwoori_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment exrwoori with eps=$eps, steps=$steps"
        # else
        #     echo "Skipping exrwoori experiment with eps=$eps, steps=$steps as output already exists"
        # fi

        # 检查exrwoori1实验的输出文件是否存在
        # if [ ! -f "$exrwoori1_output_file" ]; then
        echo "Running experiment exrwoori1 with eps=$eps, steps=$steps, dire=$dire, lp=$lp"
        python dolphins_bench_attack_exr.py --one --eps "$eps" --steps "$steps" --dire "$dire" --lp "$lp" # >> output_eps_${eps}_steps_${steps}.log 2>&1
        python tools/dolphin_evaluate.py --exp bench_attack_exrwoori1_white_${lp}_eps${eps}_steps${steps}_${dire}
        echo "Finished experiment exrwoori1 with eps=$eps, steps=$steps"
        # else
        #     echo "Skipping exrwoori1 experiment with eps=$eps, steps=$steps as output already exists"
        # fi
    done
done

# python tools/fill_table.py

