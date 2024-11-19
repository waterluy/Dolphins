#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python gen_black/dolphins_bench_attack_gen_pgd.py --eps 0.1 --steps 160 --lp linf --dire pos

