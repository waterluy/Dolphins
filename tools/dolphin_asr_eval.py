import random
import os
import argparse
import json
import pdb
import numpy as np
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tqdm import tqdm
from collections import defaultdict
from gpt_asr import GPTEvaluation
import csv

import json
import os
from collections import defaultdict
from tqdm import tqdm

NUM2WORD_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" }
NUM2WORD_CONSTANT = {str(k): v for k,v in NUM2WORD_CONSTANT.items()}
WORD2MUM_CONSTANT = {v : k for k,v in NUM2WORD_CONSTANT.items()}

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--gpt', type=str, default='gpt-3.5-turbo-0613')
    args = parser.parse_args()
    benchmark_file = "./playground/dolphins_bench/dolphins_benchmark.json"
    chatgpt_score_file = "playground/dolphins_bench/results/dolphins/dolphins_scores.json"
    result_file =  args.exp
    
    task_num = defaultdict(lambda: 0)
    task_scores = defaultdict(list)

    assert 'dolphin_output.json' in result_file
    fp_gpt = open(result_file.replace('dolphin_output.json', 'eval_asr.json'), "w")
    
    with open(benchmark_file, "r") as f:
        dolphins_bench = json.load(f) 
        dolphins_bench = [line for line in dolphins_bench]
        unique_id2label = {line['id']: line['label'] for line in dolphins_bench}
        for example in tqdm(dolphins_bench):
            task_num[example['task_name']] += 1
        
    with open(result_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [line for line in results if "drivelm" not in line['unique_id']]
        for line in results:
            if line['label'] == None: 
                line['task_name'] = line['task_name']
                continue
            line['task_name'] = line['task_name'] + "_" + line['label']
        
    print(len(results))
    
    chatgpt_eval = GPTEvaluation(gpt=args.gpt)
    total = 0
    attack_num = 0
    for idx, line in tqdm(enumerate(results)):
        total = total + 1
        unique_id = line['unique_id']
        task_name = line['task_name']
        line['pred'] = line['pred'][line['pred'].rfind("<answer>") + len("<answer>"): line['pred'].rfind("<|endofchunk|>")].lower()
        pred = line['pred']
        if pred in NUM2WORD_CONSTANT.keys(): pred = NUM2WORD_CONSTANT[pred]
        gt = line['gt']
        
        is_attack = chatgpt_eval.forward(answer=pred, GT=gt)
        if is_attack:
            attack_num = attack_num + 1
        fp_gpt.write(json.dumps({unique_id: is_attack}) + "\n")
    print(attack_num / total)
    csv_headers = ["ASR", attack_num / total]

    csv_path = result_file.replace('dolphin_output.json', 'asr.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()

    fp_gpt.close()