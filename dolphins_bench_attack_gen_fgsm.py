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

def fgsm_attack(model, vision_x, inputs, epsilon=0.001, dire='pos'):
    noise = torch.zeros_like(vision_x).to(device).half().cuda()
    noise.requires_grad = True
    loss = model(
        vision_x=(vision_x).half().cuda() + noise,
        lang_x=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        labels=None,
    )
    # print(loss.logits.shape)    # torch.Size([1, 20, 50281])
    loss = torch.norm(loss.logits, p=2)
    loss.backward()
    grad = noise.grad.detach()
    if dire == 'neg':
        noise = noise - epsilon * grad.sign()
    else:
        noise = noise + epsilon * grad.sign()
    return noise.detach()

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073] ).view(3, 1, 1)
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--dire', type=str, default='pos', choices=['pos', 'neg'])
    args = parser.parse_args()
    model, image_processor, tokenizer = load_pretrained_modoel()
    device = model.device
    model.attack = True
    BLACK_NOISE = None

    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}
    json_file = f'results/bench_attack_fgsm_white_eps{args.eps}_dire{args.dire}.json'
    with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as file:
        data = json.load(file)

    # 遍历JSON数据
    num = 0
    for entry in tqdm(data, desc='fgsm attack'):
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

        # fgsm attack
        noise = fgsm_attack(model=model, inputs=inputs, vision_x=vision_x, epsilon=args.eps, dire=args.dire)

        if BLACK_NOISE is None:
            BLACK_NOISE = noise
        else:
            BLACK_NOISE = BLACK_NOISE + noise
        num += 1

    from torchvision.utils import save_image
    BLACK_NOISE = BLACK_NOISE.detach() / num
    BLACK_NOISE = BLACK_NOISE.squeeze().mean(dim=0)
    save_image(BLACK_NOISE, f"black/dolphin_fgsm_eps{args.eps}_dire{args.dire}.png")
    BLACK_NOISE = BLACK_NOISE * image_std.to(BLACK_NOISE.device) + image_mean.to(BLACK_NOISE.device)
    BLACK_NOISE = torch.clamp(BLACK_NOISE, 0, 1)
    save_image(BLACK_NOISE, f"black/dolphin_fgsm_eps{args.eps}_dire{args.dire}_denorm.png")
        
