
#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# 获取脚本文件的路径
script_dir=$(dirname "$0")
# 获取文件夹的名字
folder_name=$(basename "$script_dir")
# 获取脚本文件的名字
# script_name=$(basename "$0" .sh)

output="wlu_outputs/${folder_name}"
mkdir -p $output

# 定义可变参数范围
eps=0.1
steps=10
exp_name="${output}/${script_name}"
json_file="test.json"

VALUES=(0.1 0.15 0.2)

for eps in "${VALUES[@]}"; do
    echo "开始运行 eps = $eps"

    exp_name="${output}/defense-eps${eps}"

    python exr/dolphins_bench_attack_pgd_white.py \
    --output $exp_name \
    --eps "$eps" \
    --steps "$steps" \
    --ckpt "ckpts/20250522/adapterRes_both_key_entropy_atten/llava_bddx/step_540/checkpoint540.pt" \
    --forward_type 15

    python tools/dolphin_evaluate.py \
    --exp ${exp_name}/dolphin_output.json \
    --api 'aihub' \
    --gpt 'gpt-3.5-turbo'

    # exp_name="${output}/attack-eps${eps}"

    # python exr/dolphins_bench_attack_pgd_white.py \
    # --output $exp_name \
    # --eps "$eps" \
    # --steps "$steps" \
    # --forward_type 0

    # python tools/dolphin_evaluate.py \
    # --exp ${exp_name}/dolphin_output.json \
    # --api 'aihub' \
    # --gpt 'gpt-3.5-turbo'
done







