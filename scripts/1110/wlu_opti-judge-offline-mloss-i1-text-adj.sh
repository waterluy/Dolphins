#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python attack/wlu_opti-judge-offline-mloss-i1.py --sup-text --sup-adj  --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-judge-offline-cos-i1-text-adj_eps0.2_iter20_query8/dolphin_output.json


