#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# python attack/wlu_opti-judge-offline-mloss-i2.py --sup-text --sup-clean --sup-adj --eps 0.2 --iter 20 --query 8 --loss cos
python tools/dolphin_evaluate.py --exp results/bench_attack_coi-opti-judge-offline-cos-i2-text-clean-adj_eps0.2_iter20_query8_lamb1-1.0_lamb2-1.0_lamb3-0.05/dolphin_output.json


