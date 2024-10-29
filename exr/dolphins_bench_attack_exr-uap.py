import sys
sys.path.append(".")
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
from tools.gpt_gen_multq import gen_multi_version

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


def get_model_inputs_prompts_for_loss(instructions, target, model,  tokenizer):
    bos_item = torch.LongTensor([tokenizer.bos_token_id])
    eos_item = torch.LongTensor([tokenizer.eos_token_id])
    bos_mask = torch.LongTensor([1])
    eos_mask = torch.LongTensor([1])
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    
    input_ids_list = []
    attention_mask_list = [] 
    labels_list = []

    for single_instruction in instructions:
        prompt = f"USER: <image> is a driving video. {single_instruction} GPT:<answer>" + target + "<|endofchunk|>"
        
        res = tokenizer(prompt, return_tensors="pt", padding="do_not_pad", truncation=True,
                        max_length=512, add_special_tokens=False).to(model.device)
        res["input_ids"] = torch.cat([bos_item, res["input_ids"].squeeze(0), eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([bos_mask, res["attention_mask"].squeeze(0), eos_mask]).unsqueeze(0)
        input_ids = res["input_ids"].cuda()
        attention_mask = res["attention_mask"].cuda()
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.eos_token] = -100
        labels[:, 0] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token the before the first <answer>
            token_idx = 0
            while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                labels[i][token_idx] = -100
                token_idx += 1
        labels = labels.to(input_ids.device)
        labels[labels == answer_token_id] = -100
        labels[labels == media_token_id] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return input_ids_list, attention_mask_list, labels_list

def get_model_inputs_images(video_path, image_processor,):
    frames = get_image(video_path)
    vision_x = torch.stack([image_processor(image) for image in frames], dim=0).unsqueeze(0).unsqueeze(0)
    assert vision_x.shape[2] == len(frames)
    # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])
    return vision_x

def get_model_inputs_prompts(instruction, model, tokenizer):
    prompt = [
        f"USER: <image> is a driving video. {instruction} GPT:<answer>"
    ]
    inputs = tokenizer(prompt, return_tensors="pt", ).to(model.device)
    # print(prompt)
    return inputs

def normalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).half().to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).half().to(tensor.device)
    return (tensor - mean) / std

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).half().to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).half().to(tensor.device)
    return tensor * std + mean

def exr_attack(model, vision_x, input_ids_list, attention_mask_list, labels_list=None, epsilon=0.001, steps=10, lp='linf', dire='pos'):
    noise = torch.zeros_like(vision_x).to(device).half().cuda()
    alpha = 2 * epsilon / steps
    denormed_vision_x = denormalize(vision_x, image_mean, image_std)
    for _ in range(steps):
        noise.requires_grad = True
        vision_x_noise = denormed_vision_x.half().cuda() + noise
        vision_x_noise = normalize(vision_x_noise, image_mean, image_std)
        if METHOD == 2:
            idx = 0
        else:
            idx = random.randrange(0, version_num)   # random.randrange(start, stop)生成一个范围内的整数，但不包括stop
        loss = model(
            vision_x=vision_x_noise,
            lang_x=input_ids_list[idx].cuda(),
            attention_mask=attention_mask_list[idx].cuda(),
            labels=labels_list[idx].cuda(),
            media_locations=None
        )[0]
        noise.grad = None
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

def exr_attack_one(model, vision_x, input_ids_list, attention_mask_list, labels_list=None, epsilon=0.001, steps=10, lp='linf', dire='pos'):
    noise = torch.zeros_like(vision_x).to(device).half().cuda()
    alpha = 2 * epsilon / steps
    denormed_vision_x = denormalize(vision_x, image_mean, image_std)
    idx = 0  # random.randrange(0, version_num)   # random.randrange(start, stop)生成一个范围内的整数，但不包括stop
    for _ in range(steps):
        noise.requires_grad = True
        vision_x_noise = denormed_vision_x.half().cuda() + noise
        vision_x_noise = normalize(vision_x_noise, image_mean, image_std)
        loss = model(
            vision_x=vision_x_noise,
            lang_x=input_ids_list[idx].cuda(),
            attention_mask=attention_mask_list[idx].cuda(),
            labels=labels_list[idx].cuda(),
            media_locations=None
        )[0]
        noise.grad = None
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

image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--lp', type=str, default='linf', choices=['l1', 'l2', 'linf'])
    parser.add_argument('--dire', type=str, default='pos', choices=['pos', 'neg'])
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--method', type=int,  choices=[2, 3, 4])
    args = parser.parse_args()
    model, image_processor, tokenizer = load_pretrained_modoel()
    device = model.device
    model.attack = True

    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}
    METHOD = int(args.method)
    houzhui = '_multi_woori'
    folder = f'results/bench_attack_m{METHOD}_white_{args.lp}_eps{args.eps}_steps{args.steps}_{args.dire}'
    os.makedirs(folder, exist_ok=True)
    json_file = os.path.join(folder, 'dolphin_output.json')
    bench_path, version_num = gen_multi_version(samples=args.samples, houzhui=houzhui)
    with open(bench_path, 'r') as file:
        data = json.load(file)

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
            assert len(entry['conversations']) == 2
            for conversation in entry['conversations']:
                if conversation['from'] == 'human':
                    instruction = conversation['value']['ori']
                    multi_version_instructions = conversation['value']['multi_version']
                elif conversation['from'] == 'gpt':
                    ground_truth = conversation['value']
            if instruction == '':
                continue

            tokenizer.eos_token_id = 50277
            tokenizer.pad_token_id = 50277

            if METHOD == 3:
                multi_version_instructions.append(instruction)

            images = get_model_inputs_images(video_path=video_path, image_processor=image_processor)
            input_ids_list, attention_mask_list, labels_list = get_model_inputs_prompts_for_loss(instructions=multi_version_instructions, target=ground_truth, model=model, tokenizer=tokenizer)

            noise = exr_attack(model=model, vision_x=images, input_ids_list=input_ids_list, attention_mask_list=attention_mask_list, labels_list=labels_list, epsilon=args.eps, steps=args.steps, lp=args.lp, dire=args.dire)

            # inference
            inputs = get_model_inputs_prompts(instruction=instruction, model=model, tokenizer=tokenizer)
            generated_tokens = model.generate(
                vision_x=images.half().cuda() + noise,
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



