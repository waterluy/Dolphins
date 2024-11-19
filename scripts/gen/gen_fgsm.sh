#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

python gen_black/dolphins_bench_attack_gen_fgsm.py --eps 0.1 --dire pos

