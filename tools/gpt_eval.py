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
        self.api_key = 'sk-I3lh2VFtgZQ45ZQv7fF0Ef29B16642F99451A5A8DfCb46D3'
        self.url = 'https://api.bianxie.ai/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model='gpt-4o-all'):
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
    
    def forward(self, answer, GT):
        prompts = "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only. "
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer
        
        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply = self.call_chatgpt(messages, max_tokens=3000)

        output += reply
        output += "\n\n"

        output = output[:-2]

        return output
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT Evaluation')
    parser.add_argument('--csv_path', type=str, default='csvfiles/dolphins_benchmark_attack_online_gpt_target.csv', help='path to the data')
    args = parser.parse_args()

    data = []
    scores = []
    eval = GPTEvaluation()
    
    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # 遍历每一行，提取 ground_truth 和 dolphins_inference
        for row in reader:
            ground_truth = row['ground_truth']
            dolphins_inference = row['dolphins_inference']
            
            success = False
            while not success:
                score = "error"
                try:
                    score = eval.forward(answer=dolphins_inference, GT=ground_truth)
                    int_score = int(score)
                except Exception as e:
                    print(e, score)
                    success = False
                else:
                    print("Success: ", int_score)
                    scores.append(int_score)
                    success = True

    avg_score = sum(scores) / len(scores)
    print("Average GPT Score: ", avg_score)

    save_path = args.csv_path.replace(".csv", "_gpt.txt")
    with open(save_path, mode='w',) as f:
        for s in scores:
            f.write(f"{s}\n")
        f.write(f"Average GPT Score: {avg_score}")
    
    