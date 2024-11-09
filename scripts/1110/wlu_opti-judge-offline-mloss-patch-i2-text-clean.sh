#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python attack/wlu_opti-judge-offline-mloss-patch-i2.py --sup-text --sup-clean --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp bench_attack_coi-opti-judge-offline-cos-patch-i2-text-clean_eps0.2_iter20_query8/dolphin_output.json


