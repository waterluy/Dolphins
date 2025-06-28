
#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

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
method=4

# 定义固定参数
dire="pos"
lp="linf"
samples=3
affine=True
json_file="test.json"

VALUES=(0.01 0.05 0.2 0.5 1.0)

for lamba in "${VALUES[@]}"; do
    echo "开始运行 lamba = $lamba" 

    exp_name="${output}/advlm-lamba${lamba}"

    python exr/dolphins_bench_attack_exr.py \
    --output $exp_name \
    --samples $samples \
    --method $method \
    --affine \
    --lp "$lp" \
    --eps "$eps" \
    --steps "$steps" \
    --ckpt "ckpts/20250622/adapter_both_key_entropy_atten-lamba${lamba}/new_checkpoint/checkpoint0.pt" \
    --forward_type 15 

    python tools/dolphin_evaluate.py \
    --exp ${exp_name}/dolphin_output.json \
    --api 'aihub' \
    --gpt 'gpt-3.5-turbo'
done







