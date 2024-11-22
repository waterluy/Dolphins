#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python vs_attack/dolphins_bench_attack_general.py --method anyattack
python tools/dolphin_evaluate.py --exp results_vs/anyattack/dolphin_output.json

