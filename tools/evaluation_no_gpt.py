import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool

import sys
sys.path.append(".")
from gpt_eval import GPTEvaluation

import csv

class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        # self.chatgpt_eval = GPTEvaluation()
        # self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        scores = sum(scores) / len(scores)
        return scores

    # def eval_chatGPT(self, data):
    #     with Pool(32) as p:  # Change the number based on your CPU cores
    #         scores = p.map(self.chatgpt_eval.forward, data)

    #     scores = list(map(float, scores))
    #     scores = sum(scores) / len(scores)
    #     return scores

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

    # def eval_match(self):
    #     outs1 = []
    #     for i in range(len(self.match["match"]["answer"])):
    #         answer = self.match["match"]["answer"][i]
    #         GT = self.match["match"]["GT"][i]
    #         _, F1_score = self.match_result(answer, GT)
    #         outs1.append(F1_score * 100)
        
    #     outs1 = sum(outs1) / len(outs1)
    #     # outs2 = self.eval_chatGPT(self.match["GPT"])

    #     # scores = (outs1 + outs2) / 2.0
    #     scores = outs1
    #     return scores

    def eval_graph(self, question):
        # check if answer in self.graph  
        question_nums = re.findall(r'\d+\.\d+', question)
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """
        answer_nums = re.findall(r'\d+\.\d+', answer)
        GT_nums = re.findall(r'\d+\.\d+', GT)
        # transform string into float
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)  
                GT_nums = np.delete(GT_nums, closest_id, axis=0) 
            else:
                false_positives += 1
            
        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        # if 0 in tag:
        #     self.accuracy["answer"].append(answer)
        #     self.accuracy["GT"].append(GT)
        # if 1 in tag:
        #     self.GPT.append((answer, GT))
        # if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
        # if 3 in tag:
        #     self.match["match"]["GT"].append(GT)
        #     self.match["match"]["answer"].append(answer)
        #     self.match["GPT"].append((answer, GT))

            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        # scores["accuracy"] = self.eval_acc()
        # scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        scores["language"] = self.eval_language()
        # scores["match"] = self.eval_match()

        return scores

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    # parser.add_argument('--csv_path', type=str, default="csvfiles/dolphins_benchmark_inference.csv", help='path to prediction file')
    # parser.add_argument('--results_path', type=str, default="./csvfiles/dolphins_benchmark_inference.txt")
    
    parser.add_argument('--csv_path', type=str, default="csvfiles/dolphins_benchmark_attack_online_gpt_target.csv", help='path to prediction file')
    parser.add_argument('--results_path', type=str, default="csvfiles/dolphins_benchmark_attack_online_gpt_target.txt")
    args = parser.parse_args()

    evaluation = evaluation_suit()

    # 打开 CSV 文件并读取内容
    with open(args.csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        
        # 遍历每一行，提取 ground_truth 和 dolphins_inference
        for row in reader:
            video_path = row['video_path']
            GT = row['ground_truth']
            task_name = row['task_name']
            predict = row['dolphins_inference']
    
            evaluation.forward(task_name, predict, GT)

    output = evaluation.evaluation()

    fw = open(args.results_path, 'w')

    # print("accuracy score: ", output["accuracy"])

    # fw.write("accuracy score: "+str(output["accuracy"])+'\n')
    # print("chatgpt score: ", output["chatgpt"])
    # print("match score: ", output["match"])
    # fw.write("match score: "+str(output["match"])+'\n')
    print("language score: ", output["language"])
    fw.write("language score: "+str(output["language"])+'\n')
    
    # Normalize to 0-1 and combine the scores: chatgpt, language, match, accuracy
    scores = []
    weights = [1.0]
    
    # chatGPT
    # score = output["chatgpt"] / 100.
    # scores.append(score)

    # language
    score = 0
    for idx, key in enumerate(output["language"].keys()):
        if idx < 4:
            score += output["language"][key] / 4. / 3.
        elif idx == 4:
            score += output["language"][key] / 3. 
        else:
            score += output["language"][key] / 10. / 3.

    scores.append(score)

    assert len(scores) == len(weights)
    final_score = sum([x * y for x, y in zip(scores, weights)])
    print("final score: ", final_score)
    fw.write("final score: "+str(final_score)+'\n')
    fw.close()
    print(f"write to {args.results_path}")