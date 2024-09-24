import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Union
from PIL import Image
import mimetypes
import requests
import numpy as np
import random
import copy
import csv
import warnings
warnings.filterwarnings("ignore")

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

# ------------------- 图像和视频处理函数 -------------------
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
    if "://" not in url:  # 本地文件
        content_type = get_content_type(url)
    else:  # 远程URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # 本地文件
            return Image.open(url)
        else:  # 远程URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # 本地文件
            video_path = url
        else:  # 远程URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # 如果是下载的临时视频文件，删除它
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")

def load_pretrained_model():
    peft_config, peft_model_id = None, None
    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="/home/beihang/zxw/Dolphins/mpt-7b", # anas-awadalla/mpt-7b
        tokenizer_path="/home/beihang/zxw/Dolphins/mpt-7b",  # anas-awadalla/mpt-7b
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
    )

    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.half().cuda()

    return model, image_processor, tokenizer

def get_model_inputs(video_path, instruction, model, image_processor, tokenizer, conversation_history=None):
    frames = get_image(video_path)
    vision_x = torch.stack([image_processor(image) for image in frames], dim=0).unsqueeze(0).unsqueeze(0)
    assert vision_x.shape[2] == len(frames)
    if conversation_history is not None:
        new_conversation_history = copy.deepcopy(conversation_history)

    if conversation_history is not None:
        if len(new_conversation_history) == 0:
            new_conversation_history.append(f"USER: <image> is a driving video. {instruction} GPT:<answer>")
        else:
            new_conversation_history.append(f"USER: {instruction} GPT:<answer>")
        prompt = "\n".join(new_conversation_history)
    else:
        prompt = f"USER: <image> is a driving video. {instruction} GPT:<answer>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   
    return vision_x, inputs

def read_instruction(instruction_path):
    with open(instruction_path, "r") as f:
        data = json.load(f)
    instructions = []
    for key, value in data.items():
        if "conversation" in value:
            for convo in value['conversation']:
                if 'prompt' in convo:
                    instructions.append(convo['prompt'])
    #                 print(convo['prompt'])
    # print(len(instructions))
    return instructions

if __name__ == "__main__":
    video_path = "./playground/videos/1.mp4"
    conversation_history = []

    model, image_processor, tokenizer = load_pretrained_model()
    tokenizer.eos_token_id = 50277
    tokenizer.pad_token_id = 50277

    instructions = [
        "Who are you?",
        "How is the weather?",
        "What should you do?",
        "Why?"
    ]
    instructions += read_instruction("./captions_BDDX_clean.json")

    # 创建CSV文件，设置表头
    with open('./csvfiles/tmp.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["instruction", "dolphins answer"])  # 写入表头
        for instruction in instructions:
            # 保留 conversation_history
            # vision_x, inputs = get_model_inputs(video_path, instruction, model, image_processor, tokenizer, conversation_history)
            # 不保留 conversation_history 
            vision_x, inputs = get_model_inputs(video_path, instruction, model, image_processor, tokenizer)
            
            generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}

            generated_tokens = model.generate(
                vision_x=vision_x.half().cuda(),
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

            if len(conversation_history) == 0:
                conversation_history.append(generated_text[0])
            else:
                conversation_history[0] = generated_text[0]

            print(f"instruction: {instruction}\ndolphins answer: {content_after_last_answer}\n\n")
            # 将指令和生成的答案写入CSV文件
            writer.writerow([instruction, content_after_last_answer])
