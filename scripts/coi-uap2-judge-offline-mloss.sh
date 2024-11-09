#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python attack/wlu_opti-uap2-judge-offline-mloss.py --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-uap2-judge-offline-cos_eps0.2_iter20_query8/dolphin_output.json


python attack/wlu_opti-uap2-judge-offline-mloss.py --eps 0.2 --iter 20 --query 8 --loss kl
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-uap2-judge-offline-kl_eps0.2_iter20_query8/dolphin_output.json

