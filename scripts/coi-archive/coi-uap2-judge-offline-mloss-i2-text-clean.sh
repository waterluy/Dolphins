#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

# python attack/wlu_opti-uap2-judge-offline-mloss-i2.py --sup-text --sup-clean --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-uap2-judge-offline-cos-i2-text-clean_eps0.2_iter20_query8/dolphin_output.json


