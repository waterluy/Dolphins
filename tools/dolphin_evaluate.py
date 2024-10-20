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
from gpt_eval import GPTEvaluation
import csv

def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for idx, cap in enumerate(caps):
            captions_for_image[key].append({
                # "image_id": key
                # "id": idx,
                "caption": cap
            })
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions

def evaluate_from_file(key_to_pred, key_to_refs, no_spice=True):
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)
    if no_spice:
        scorers = [Bleu(n=4), Rouge(), Meteor()]
    else:
        scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]

    results = np.empty((len(key_to_refs.keys()), 9))  # save results to csv
    index = 0

    output = {
        'SPIDEr': 0,
        'Bleu': None,
        'Rouge': 0,
        'METEOR': 0,
        'CIDEr': 0,
        'SPICE': 0
    }
    for scorer in scorers:
        score, scores = scorer.compute_score(key_to_refs, key_to_pred)
        method = scorer.method()
        output[method] = score
        if method == "Bleu":
            for n in range(4):
                results[:, index] = scores[n]
                index += 1
        else:
            if method == 'SPICE':
                tmp_scores = []
                for img in scores:
                    img_score = np.array([x['f'] for x in img.values()])
                    tmp_scores.append(img_score[~np.isnan(img_score)].mean())
                tmp_scores = np.array(tmp_scores)
                results[:, index] = tmp_scores
                index += 1
            else:
                results[:, index] = scores
                index += 1
            if method in ["CIDEr", "SPICE"]:
                output["SPIDEr"] += score

    results[:, -1] = (results[:, -2] + results[:, -3]) / 2  # SPIDEr
    output["SPIDEr"] /= 2

    return output


import json
import os
from collections import defaultdict
from tqdm import tqdm

NUM2WORD_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" }
NUM2WORD_CONSTANT = {str(k): v for k,v in NUM2WORD_CONSTANT.items()}
WORD2MUM_CONSTANT = {v : k for k,v in NUM2WORD_CONSTANT.items()}


class Evaluation():
    def __init__(self, chatgpt_scores=None):
        if chatgpt_scores is not None:
            self.chatgpt_scores = chatgpt_scores
        else:
            self.chatgpt_scores = None
            self.chatgpt_eval = GPTEvaluation()

    def eval_acc(self, unique_id, answer, GT):
        if "or" in GT: 
            labels = GT.lower().split("or")
        else:
            labels = [GT.lower()]
        for gt in labels:
            if gt in answer: return 1
        return 0

    def eval_chatGPT(self, unique_id, answer, GT):
        if self.chatgpt_scores is not None:
            return self.chatgpt_scores[unique_id]
        else:
            scores = self.chatgpt_eval.forward(answer=answer, GT=GT)
            scores = float(scores)
        return scores

    def eval_language(self, answer, GT):
        """
        return the dict evaluation results
        """
        results = evaluate_from_file(answer, GT)
        results_gen_dict = {
            "Bleu-1": results['Bleu'][0],
            "Bleu-2": results['Bleu'][1],
            "Bleu-3": results['Bleu'][2],
            "Bleu-4": results['Bleu'][3],
            "Rouge": results['Rouge'],
            "METEOR": results['METEOR'],
        }
        return results_gen_dict

    def forward(self, unique_id, answer, GT, label=None):
        scores = {}
        # if label is not None and unique_id in self.chatgpt_scores.keys():
        scores["chatgpt"] = self.eval_chatGPT(unique_id, answer, GT) / 100
        if label is not None:
            scores["accuracy"] = self.eval_acc(unique_id, answer, label)
        else:
            scores["accuracy"] = scores["chatgpt"]
            
        # elif label is not None:
        #     scores["accuracy"] = self.eval_acc(unique_id, answer, label)
        #     scores["chatgpt"] = scores["accuracy"]
            
        # elif unique_id in self.chatgpt_scores.keys():
        #     scores["chatgpt"] = self.eval_chatGPT(unique_id, answer, GT)
        #     scores["accuracy"] = scores["chatgpt"]
            
        # scores["language"] = self.eval_language(answer, GT)
        return scores
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None)
    args = parser.parse_args()
    benchmark_file = "./playground/dolphins_bench/dolphins_benchmark.json"
    # result_file = "results/dolphins_benchmark_attack_online_gpt_target_0.json"
    # result_file = 'playground/dolphins_bench/results/dolphins/dolphins_results.json'
    chatgpt_score_file = "playground/dolphins_bench/results/dolphins/dolphins_scores.json"
    result_file =  os.path.join('results', args.exp, 'dolphin_output.json')
    
    task_num = defaultdict(lambda: 0)
    task_scores = defaultdict(list)

    fp = open(result_file.replace('dolphin_output.json', 'eval_log.txt'), "w")
    
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
    fp.write(str(len(results)) + "\n")
   
    # with open(chatgpt_score_file, "r") as f:
    #     chatgpt_scores = [json.loads(line) for line in f.readlines()]
    #     chatgpt_scores = {line['unique_id']: float(line['scores'].split("\n")[0].strip(" ")) for line in chatgpt_scores}
    chatgpt_scores = None
    
    evaluation = Evaluation(chatgpt_scores=chatgpt_scores)
    outputs = defaultdict(lambda: {"accuracy": [], "chatgpt": [], "language": []})
    cnt = 0
    for idx, line in tqdm(enumerate(results)):
        unique_id = line['unique_id']
        task_name = line['task_name']
        line['pred'] = line['pred'][line['pred'].rfind("<answer>") + len("<answer>"): line['pred'].rfind("<|endofchunk|>")].lower()
        pred = line['pred']
        if pred in NUM2WORD_CONSTANT.keys(): pred = NUM2WORD_CONSTANT[pred]
        gt = line['gt']
        try:
            label = unique_id2label[unique_id]
        except:
            label = None
        
        res = evaluation.forward(unique_id, pred, gt, label) 
        for key in outputs[task_name].keys():
            if key in res:
                outputs[task_name][key].append(res[key])
                

    final_scores = {
        "weather": {},
        "traffic_light": {},
        "timeofday": {}, 
        "scene": {},
        "open_voc_object": {},
        "detailed_description": {}
    }

    task_instance_num = {}
                
    for task_name in outputs.keys():
        print(f"========{task_name}========")
        fp.write(f"========{task_name}========" + "\n")
        output = outputs[task_name]
        preds = {line['unique_id']: [line['pred']] for line in results if line['task_name'] == task_name}
        gts = {line['unique_id']: [line['gt']] for line in results if line['task_name'] == task_name}
        
        task_instance_num[task_name] = len(list(gts.keys()))
        assert len(preds.keys()) == len(gts.keys())
        language_score = evaluation.eval_language(preds, gts)
      
        print("language: ", language_score)
        fp.write("language: "+str(language_score)+"\n")
        if len(output["accuracy"]) != 0:
            output["accuracy"] = sum(output["accuracy"]) / len(output["accuracy"])
            print("accuracy: ", output["accuracy"])
            fp.write("accuracy: "+str(output["accuracy"])+"\n")
        else:
            output["accuracy"] = 0.0
        if len(output["chatgpt"]) != 0:
            output["chatgpt"] = sum(output["chatgpt"]) / len(output["chatgpt"])
            print("chatgpt: ", output["chatgpt"])
            fp.write("chatgpt: "+str(output["chatgpt"])+"\n")
        else:
            output["chatgpt"] = 0.0
        
        scores = []
        weights = [0.4, 0.4, 0.2]
        
        # chatGPT
        score = output["chatgpt"]
        scores.append(score)

        # language
        score = 0
        for idx, key in enumerate(language_score.keys()):
            if idx < 4:
                score += language_score[key] / 4. / 3.
            else:
                score += language_score[key] / 3. 
        scores.append(score)
        
        # accuracy
        score = output["accuracy"]
        scores.append(score)
        final_score = sum([x * y for x, y in zip(scores, weights)])
        
        for task in final_scores.keys():
            if task in task_name:
                final_scores[task][task_name] = final_score * 100
    
    csv_data = []
    csv_headers = ["task_name", "weighted_score"]
    score_sum = 0
    task_num = 0
    for task in final_scores.keys():
        scores, weights = [], []
        for sub_task in final_scores[task].keys():
            scores.append(final_scores[task][sub_task])
            weights.append(task_instance_num[sub_task])

        weighted_average = np.average(np.array(scores), weights=np.array(weights))
        print(
            f"{task} weighted score: {weighted_average}"
        )
        fp.write(f"{task} weighted score: {weighted_average}\n")
        csv_data.append({
            "task": task,
            "score": weighted_average
        })
        score_sum += weighted_average
        task_num += 1

    csv_path = os.path.join('results', args.exp, 'bench_score.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_data)
        writer.writerow({
            "task_name": 'avg',
            "weighted_score": score_sum / task_num
        })

    print(final_scores)
    fp.write(str(final_scores)+"\n")
    print(task_instance_num)
    fp.write(str(task_instance_num)+"\n")
    fp.close()

 