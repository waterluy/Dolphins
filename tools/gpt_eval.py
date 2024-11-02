import pickle
import pdb
import numpy as np
import torch
import json
import argparse
from multiprocessing import Pool
from openai import OpenAI
import requests
import csv

class GPTEvaluation:
    def __init__(self):
        with open("tools/api.json", 'r') as file:
            data = json.load(file)["eval"]
        self.api_key = data['api_key']
        self.url = data['base_url']
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model='gpt-3.5-turbo'):    # gpt-3.5-turbo-0613 gpt-4o-all
        data = {
            "model": model,
            "messages": chatgpt_messages,
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        reply = response.json()['choices'][0]['message']['content']
        return reply
    
    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})
        
        return messages
    
    def GPT_eval(self, answer, GT):
        prompts = "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only. "
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer
        
        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply = self.call_chatgpt(messages, max_tokens=3000)

        output += reply
        output += "\n\n"

        output = output[:-2]

        return output
    
    def forward(self, answer, GT) -> int:
        success = False
        if answer == '':
            return 0
        while not success:
            score = "error"
            try:
                score = self.GPT_eval(answer=answer, GT=GT)
                int_score = int(score)
            except Exception as e:
                print(e, score)
                success = False
            else:
                print("Success: ", int_score)
                success = True

        return int_score
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT Evaluation')
    parser.add_argument('--json-path', type=str, default='results/dolphins_benchmark_attack_online_gpt_target_0.json', help='path to the data')
    args = parser.parse_args()

    data = []
    scores = []
    eval = GPTEvaluation()
    
    with open(args.json_path, mode='r', newline='', encoding='utf-8') as file:
        lines = [line for line in file]
        # 遍历每一行，提取 ground_truth 和 dolphins_inference
        for row in lines:
            ground_truth = row['gt']
            dolphins_inference = row['pred']
            
            int_score = eval.forward(answer=dolphins_inference, GT=ground_truth)
            scores.append(int_score)

    avg_score = sum(scores) / len(scores)
    print("Average GPT Score: ", avg_score)

    save_path = args.json_path.replace(".json", "_gpt1.txt")
    with open(save_path, mode='w',) as f:
        for s in scores:
            f.write(f"{s}\n")
        f.write(f"Average GPT Score: {avg_score}")
    
    