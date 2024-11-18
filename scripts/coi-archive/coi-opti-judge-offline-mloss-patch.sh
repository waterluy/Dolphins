#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

python attack/wlu_opti-judge-offline-mloss-patch.py --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-judge-offline-cos-patch_eps0.2_iter20_query8/dolphin_output.json

# python attack/wlu_opti-judge-offline-mloss-patch.py --eps 0.2 --iter 20 --query 8 --loss kl
# python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-judge-offline-kl-patch_eps0.2_iter20_query8/dolphin_output.json

