#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

lamb1=0.0
lamb2=0.75
lamb3=0.05
eps=0.2

python attack/physical_attack.py --eps $eps --iter 160 --query 1 --loss cos --lamb1 $lamb1 --lamb2 $lamb2 --lamb3 $lamb3
