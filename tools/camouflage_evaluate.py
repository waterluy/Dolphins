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
import csv
from gpt_eval_aihub import GPTEvaluationAihub as GPTEvaluation
import language_evaluation
import re


class Evaluation():
    def __init__(self, folder, gpt):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "METEOR", "CIDEr"])
        self.chatgpt_eval = GPTEvaluation(gpt)
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.idx2GPT = []
        self.idx2matchGPT = []
        self.folder = folder

    def eval_chatGPT(self, data, idxs, file_name="GPT.json"):
        # with Pool(32) as p:  # Change the number based on your CPU cores
        #     scores = p.map(self.chatgpt_eval.forward, data)
        json_data = {}
        scores = []
        for d, idx in zip(data, idxs):
            score = self.chatgpt_eval.forward(answer=d[0], GT=d[1])
            float_score = float(score)
            scores.append(float_score)
            json_data[idx] = score
        with open(os.path.join(self.folder, file_name), "w") as file:
            json.dump(json_data, file, indent=4)

        # scores = list(map(float, scores))
        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def forward(self, answer, GT, idx):
        self.GPT.append((answer, GT))
        self.idx2GPT.append(idx)
        self.language["GT"].append(GT)
        self.language["answer"].append(answer)

            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["chatgpt"] = self.eval_chatGPT(self.GPT, self.idx2GPT, "GPT.json")
        scores["language"] = self.eval_language()

        return scores

    
if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--gpt', type=str, default='gpt-4o-all')
    parser.add_argument('--api', type=str, default='aihub', choices=['bianxie', 'aihub', 'aihub1'])
    args = parser.parse_args()
    
    result_file =  args.exp
    assert 'dolphin_output.json' in result_file
    fp = open(result_file.replace('dolphin_output.json', 'eval_log.txt'), "w")
    
    with open(result_file, 'r') as f :#, \    
        results = [json.loads(line) for line in f.readlines()]
        results = [line for line in results if "drivelm" not in line['unique_id']]
    print(len(results))
    fp.write(str(len(results)) + "\n")

    evaluation = Evaluation(folder=os.path.dirname(result_file), gpt=args.gpt)
    outputs = defaultdict(lambda: {"accuracy": [], "chatgpt": [], "language": []})
    for idx, line in tqdm(enumerate(results)):
        unique_id = line['unique_id']
        line['pred'] = line['pred'][line['pred'].rfind("<answer>") + len("<answer>"): line['pred'].rfind("<|endofchunk|>")].lower()
        pred = line['pred']
        gt = line['gt']
        evaluation.forward(pred, gt, unique_id)
        

    output = evaluation.evaluation()
    
    print("chatgpt score: ", output["chatgpt"])
    print("language score: ", output["language"])
    fp.write(f"chatgpt score: {output['chatgpt']}\n")
    fp.write(f"language score: {output['language']}\n")
    
    csv_path = result_file.replace('dolphin_output.json', 'camouflage_score.csv')
    fw_csv = open(csv_path, 'w')
    header = ['final', 'chatgpt', 'BLEU4', 'ROUGEL', 'METEOR', 'language', 'lang.bleu1', 'lang.bleu2', 'lang.bleu3', 'lang.bleu4', 'lang.rougeL', 'lang.cider']
    csv_writer = csv.DictWriter(fw_csv, fieldnames=header)
    csv_writer.writeheader()
    csv_data = {}
    
    # chatGPT
    score = output["chatgpt"] / 100.
    csv_data["chatgpt"] = score

    csv_data['lang.bleu1'] = output["language"]['val/Bleu_1']
    csv_data['lang.bleu2'] = output["language"]['val/Bleu_2']
    csv_data['lang.bleu3'] = output["language"]['val/Bleu_3']
    csv_data['lang.bleu4'] = output["language"]['val/Bleu_4']
    csv_data['lang.rougeL'] = output["language"]['val/ROUGE_L']
    csv_data['lang.cider'] = output["language"]['val/CIDEr']
    csv_data['BLEU4'] = output["language"]['val/Bleu_4']
    csv_data['ROUGEL'] = output["language"]['val/ROUGE_L']
    csv_data['METEOR'] = output["language"]['val/METEOR']
    
    csv_data["language"] = (output["language"]['val/Bleu_4'] * 100 + output["language"]['val/ROUGE_L'] * 100 + output["language"]['val/METEOR'] * 100) / 3.0

    
    # language
    # score = 0
    # for idx, key in enumerate(output["language"].keys()):
    #     if idx < 4:
    #         score += output["language"][key] / 4. / 3.
    #     elif idx == 4:
    #         score += output["language"][key] / 3. 
    #     else:
    #         score += output["language"][key] / 10. / 3.
    # csv_data["language"] = score

    csv_writer.writerow(csv_data)
    fw_csv.close()
    print(f"write to {csv_path}")

