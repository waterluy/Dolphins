import pickle
import random
import numpy as np
import os
import json
import argparse
from multiprocessing import Pool
from openai import OpenAI


class GPT:
    def __init__(self):
        with open("tools/api.json", 'r') as file:
            data = json.load(file)["gen_muti"]
        self.client = OpenAI(base_url=data["base_url"],api_key=data["api_key"])

    # 调用 gpt，生成回复
    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
        response = self.client.chat.completions.create(
            model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
        )
        # print(response)
        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return reply, total_tokens # 返回回复内容 和 总token数
    
    # 准备发给gpt的消息格式
    def prepare_chatgpt_message(self, prompt):
        system_message = "A worker who generates a new sentence based on one or more given sentences that is significantly different in text but has similar semantics."
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})
        
        return messages
    
    def generate(self, data): 
        prompts = "Hello, I will give you one or more sentences and ask you to give me a sentence that has the same meaning but uses different vocabulary with them. Please give the new sentence directly without any additional information. "
        prompts += "These sentences are given in tuple form as follows: \n"

        prompts = prompts + data + "\n"

        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply, total_tokens = self.call_chatgpt(messages, max_tokens=3000)

        output += reply
        output += "\n\n"

        output = output[:-2]

        # print(output)

        return output

    def forward(self, data):
        success = False
        while not success:
            new_version = None
            try:
                new_version = self.generate(data)
            except Exception as e:
                print(e, new_version)
                success = False
            else:
                success = True

        return new_version


def gen_multi_version(samples=5, bench_path='playground/dolphins_bench/dolphins_benchmark.json'):
    output_path = bench_path.replace('.json', '_multi.json')
    if os.path.exists(output_path):
        return output_path
    
    with open(bench_path, 'r') as file:
        data = json.load(file)

    gpt = GPT()

    for entry in data:
        prompts = []
        instruction = entry['conversations'][0]['value']
        prompts.append(instruction)
        
        for _ in range(samples):
            new_version = gpt.forward(str(prompts))
            prompts.append(new_version)
        
        random.shuffle(prompts)
        entry['conversations'][0]['value'] = {
            'ori': instruction,
            'multi_version': prompts
        }

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
    return output_path


if __name__ == "__main__":
    _ = gen_multi_version()