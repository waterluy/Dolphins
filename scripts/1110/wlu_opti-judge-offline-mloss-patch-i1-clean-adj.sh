#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python attack/wlu_opti-judge-offline-mloss-patch-i1.py --sup-clean --sup-adj --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp bench_attack_coi-opti-judge-offline-cos-patch-i1-clean-adj_eps0.2_iter20_query8/dolphin_output.json


