#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

method=sga

python vs_attack/dolphins_bench_attack_general.py --method $method
python tools/dolphin_evaluate.py --exp results_vs/$method/dolphin_output.json


