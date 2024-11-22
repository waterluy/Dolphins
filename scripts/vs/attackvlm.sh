#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python vs_attack/dolphins_bench_attack_attackvlm.py --eps 0.1
python tools/dolphin_evaluate.py --exp results_vs/attackvlm0.1/dolphin_output.json


