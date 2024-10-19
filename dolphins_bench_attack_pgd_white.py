import json
import csv
import random

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Union
from PIL import Image
import mimetypes

import cv2

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import LlamaTokenizer, CLIPImageProcessor

from configs.dataset_config import DATASET_CONFIG
from configs.lora_config import openflamingo_tuning_config, otter_tuning_config

from mllm.src.factory import create_model_and_transforms
from mllm.otter.modeling_otter import OtterConfig, OtterForConditionalGeneration

from huggingface_hub import hf_hub_download
from peft import (
    get_peft_model,
    LoraConfig,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------
def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


def load_pretrained_modoel():
    peft_config, peft_model_id = None, None
    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b", # anas-awadalla/mpt-7b
        tokenizer_path="anas-awadalla/mpt-7b",  # anas-awadalla/mpt-7b
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
    )

    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.half().cuda()

    return model, image_processor, tokenizer


def get_model_inputs(video_path, instruction, model, image_processor, tokenizer):
    frames = get_image(video_path)
    vision_x = torch.stack([image_processor(image) for image in frames], dim=0).unsqueeze(0).unsqueeze(0)
    assert vision_x.shape[2] == len(frames)
    prompt = [
        f"USER: <image> is a driving video. {instruction} GPT:<answer>"
    ]
    inputs = tokenizer(prompt, return_tensors="pt", ).to(model.device)
   
    # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])
    # print(prompt)

    return vision_x, inputs

def pgd_attack(model, vision_x, inputs, epsilon=0.001, steps=10, lp='linf', dire='pos'):
    noise = torch.zeros_like(vision_x).to(device).half().cuda()
    alpha = epsilon / steps
    for _ in range(steps):
        noise.requires_grad = True
        loss = model(
            vision_x=(vision_x).half().cuda() + noise,
            lang_x=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            labels=None,
        )
        loss = torch.norm(loss.logits, p=2)
        loss.backward()
        grad = noise.grad.detach()
        if lp == 'linf':
            delta = grad.sign()
        elif lp == 'l1':
            delta = grad / torch.norm(grad, p=1)
        elif lp == 'l2':
            delta = grad / torch.norm(grad, p=2)
        else:
            raise ValueError('lp must be linf, l1 or l2')
        if dire == 'neg':
            noise = noise - alpha * delta
        else:
            noise = noise + alpha * delta
        noise = noise.detach()
    return noise.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--lp', type=str, default='linf', choices=['l1', 'l2', 'linf'])
    parser.add_argument('--dire', type=str, default='pos', choices=['pos', 'neg'])
    args = parser.parse_args()
    model, image_processor, tokenizer = load_pretrained_modoel()
    device = model.device
    model.attack = True

    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}
    json_file = f'results/bench_attack_pgd_white_{args.lp}_eps{args.eps}_steps{args.steps}_{args.dire}.json'
    with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as file:
        data = json.load(file)
    # random.shuffle(data)

    with open(json_file, 'w') as file:
        # 遍历JSON数据
        for entry in data:
            instruction = ''
            ground_truth = ''
            unique_id = entry["id"]
            label = entry['label']
            video_path = entry['video_path'][entry['video_path'].find('/')+1:]
            task_name = entry['task_name']
            # 从conversations中提取human的value和gpt的value
            for conversation in entry['conversations']:
                if conversation['from'] == 'human':
                    instruction = conversation['value']
                elif conversation['from'] == 'gpt':
                    ground_truth = conversation['value']
            if instruction == '':
                continue

            tokenizer.eos_token_id = 50277
            tokenizer.pad_token_id = 50277

            vision_x, inputs = get_model_inputs(video_path, instruction, model, image_processor, tokenizer)
            # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])

            # pgd attack
            noise = pgd_attack(model=model, inputs=inputs, vision_x=vision_x, epsilon=args.eps, steps=args.steps, lp=args.lp, dire=args.dire)

            # inference
            generated_tokens = model.generate(
                vision_x=vision_x.half().cuda() + noise,
                lang_x=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                num_beams=3,
                **generation_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            generated_text = tokenizer.batch_decode(generated_tokens)
            last_answer_index = generated_text[0].rfind("<answer>")
            content_after_last_answer = generated_text[0][last_answer_index + len("<answer>"):]
            
            print(f"\n{video_path}\n")
            print(f"\n\ninstruction: {instruction}\ndolphins answer: {content_after_last_answer}\n\n")
            # 写入json行数据
            file.write(
                json.dumps({
                    "unique_id": unique_id,
                    "task_name": task_name,
                    "pred": content_after_last_answer,
                    "gt": ground_truth,
                    "label": label
                }) + "\n"
            )



