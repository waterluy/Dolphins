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
from tqdm import tqdm
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
import torch.nn.functional as F
import numpy as np
import clip
from torchvision import transforms

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

def loss_function(predicted_output, target_output):
    """
    计算模型输出与目标输出之间的损失。
    
    参数：
    predicted_output: 模型生成的文本输出
    target_output: 目标文本输出(ground truth)
    
    返回：
    loss: 计算得到的损失值
    """
    predicted_text_features = model_clip.encode_text(clip.tokenize(predicted_output, truncate=True).to(device)).to(device)
    predicted_text_features_normed = F.normalize(predicted_text_features, dim=-1)
    target_text_features = model_clip.encode_text(clip.tokenize(target_output, truncate=True).to(device)).to(device)
    target_text_features_normed = F.normalize(target_text_features, dim=-1)
    
    # print(predicted_text_features_normed.shape, target_text_features_normed.shape)
    # print(torch.cosine_similarity(predicted_text_features_normed, target_text_features_normed, eps=1e-8))
    # 使用交叉熵损失函数计算损失
    loss = torch.cosine_similarity(predicted_text_features_normed, target_text_features_normed, eps=1e-8).mean()  # 最大化相似度
    
    return loss.detach()

def compute_loss(model, vision_x, inputs, target_output):
    # 获取模型的输出

    generated_tokens = model.generate(
        vision_x=vision_x.half().cuda(),
        lang_x=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        num_beams=3,
        **generation_kwargs,
    )
    
    generated_text = tokenizer.batch_decode(generated_tokens)
    # 提取答案并计算损失
    last_answer_index = generated_text[0].rfind("<answer>")
    content_after_last_answer = generated_text[0][last_answer_index + len("<answer>"):].strip()
    
    # 计算损失（这里可以根据需要自定义损失函数）
    loss = loss_function(content_after_last_answer, target_output)
    
    return loss

def estimate_gradient(model, vision_x, inputs, target_output, epsilon=1e-4):
    perturbation = torch.zeros_like(vision_x)
    # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])
    random_c = []
    random_h = []
    random_w = []
    for n in range(vision_x.shape[2]):
        random_c.append(torch.randint(0, vision_x.shape[3], (1,)).item())
        random_h.append(torch.randint(0, vision_x.shape[4], (1,)).item())
        random_w.append(torch.randint(0, vision_x.shape[5], (1,)).item())
        perturbation[:, :, n, random_c[n], random_h[n], random_w[n]] = 1
    perturbation = perturbation.to(device) * epsilon

    # 所有图像可以一起计算梯度
    # 计算正向扰动
    perturbed_input_plus = vision_x + perturbation
    loss_plus = compute_loss(model, perturbed_input_plus, inputs, target_output)
    
    # 计算反向扰动
    perturbed_input_minus = vision_x - perturbation
    loss_minus = compute_loss(model, perturbed_input_minus, inputs, target_output)
    
    # 估计梯度
    grad = (loss_plus - loss_minus) / (2 * epsilon)
    gradient = torch.zeros_like(vision_x).to(device)
    for n in range(vision_x.shape[2]):
        gradient[:, :, n, random_c[n], random_h[n], random_w[n]] = grad
    
    return gradient.detach()

def estimate_gradient_newt(model, vision_x, inputs, target_output, epsilon=1e-4):
    perturbation = torch.zeros_like(vision_x)
    # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])
    random_c = []
    random_h = []
    random_w = []
    for n in range(vision_x.shape[2]):
        random_c.append(torch.randint(0, vision_x.shape[3], (1,)).item())
        random_h.append(torch.randint(0, vision_x.shape[4], (1,)).item())
        random_w.append(torch.randint(0, vision_x.shape[5], (1,)).item())
        perturbation[:, :, n, random_c[n], random_h[n], random_w[n]] = 1
    perturbation = perturbation.to(device) * epsilon
    loss = compute_loss(model, vision_x, inputs, target_output)

    # 所有图像可以一起计算梯度
    # 计算正向扰动
    perturbed_input_plus = vision_x + perturbation
    loss_plus = compute_loss(model, perturbed_input_plus, inputs, target_output)
    
    # 计算反向扰动
    perturbed_input_minus = vision_x - perturbation
    loss_minus = compute_loss(model, perturbed_input_minus, inputs, target_output)
    
    # 估计梯度
    grad = (loss_plus - loss_minus) / (2 * epsilon)
    gradient = torch.zeros_like(vision_x).to(device)
    for n in range(vision_x.shape[2]):
        gradient[:, :, n, random_c[n], random_h[n], random_w[n]] = grad
    h = (loss_plus - 2 * loss + loss_minus) / (epsilon ** 2)
    hessian = torch.zeros_like(vision_x).to(device)
    for n in range(vision_x.shape[2]):
        hessian[:, :, n, random_c[n], random_h[n], random_w[n]] = h
    
    return grad.detach(), hessian.detach()

def adam_optimize(noise, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用 Adam 优化器更新噪声。
    
    参数：
    - noise: 当前噪声
    - grad: 梯度值
    - m: 一阶动量估计
    - v: 二阶动量估计
    - t: 当前迭代步数
    - lr: 学习率 (默认为 0.01)
    - beta1: 一阶动量衰减系数 (默认为 0.9)
    - beta2: 二阶动量衰减系数 (默认为 0.999)
    - epsilon: 防止除零的小数 (默认为 1e-8)
    
    返回值：
    - 更新后的噪声
    - 更新后的一阶动量 m
    - 更新后的二阶动量 v
    """
    # 更新一阶和二阶动量
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    # 计算偏差修正后的 m_hat 和 v_hat
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # 更新噪声
    noise = noise - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return noise, m, v

def newton_optimize(noise, grad, hessian, lr=1e-3):
    """
    使用牛顿法优化噪声。

    参数：
    - noise: 当前噪声
    - grad: 当前梯度
    - hessian: 当前 Hessian
    - lr: 学习率，默认值为 1e-3

    返回：
    - 更新后的噪声
    """
    # 计算 Hessian 的符号，创建布尔数组
    positive_hessian = hessian > 0

    # 创建 delta_star 数组
    delta_star = torch.zeros_like(noise)

    # 处理负 Hessian 的情况
    delta_star[~positive_hessian] = -lr * grad[~positive_hessian]

    # 处理正 Hessian 的情况
    delta_star[positive_hessian] = -lr * (grad[positive_hessian] / hessian[positive_hessian])

    # 更新噪声
    noise += delta_star
    
    return noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT Evaluation')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--opt', type=str, default='none', choices=["adam", "newt", "none"])
    args = parser.parse_args()
    LR = args.lr
    ITER = args.iter

    model, image_processor, tokenizer = load_pretrained_modoel()
    device = model.device
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=model.device) 
    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}

    with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as file:
        data = json.load(file)
    # random.shuffle(data)
    # 初始化进度条，设置position=0以确保它在最底部
    progress_bar = tqdm(total=len(data), position=0)

    with open(f'results/dolphins_benchmark_attack_zoo_{LR}_{ITER}_{args.opt}.json', 'w') as file:
        iter_num = 0
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

            # 初始化噪声
            noise = torch.zeros_like(vision_x).to(device)
            # 初始化动量和时间步数
            m = torch.zeros_like(noise).to(device)
            v = torch.zeros_like(noise).to(device)
            for iter_num in range(ITER):  # num_iterations 代表迭代次数
                print(f"Iteration: {iter_num}")
                for _ in range(args.samples):
                    # 计算目标输出
                    target_output = ground_truth  # 或者使用其他目标
                    
                    if args.opt == 'adam':
                        # 估计梯度
                        grad = estimate_gradient(model, vision_x + noise, inputs, target_output)
                        # 使用 Adam 优化噪声
                        noise, m, v = adam_optimize(noise, grad, m, v, iter_num+1, lr=LR)
                    elif args.opt == 'newt':
                        # 估计梯度
                        grad, hessian = estimate_gradient_newt(model, vision_x + noise, inputs, target_output)
                        # 使用牛顿法优化噪声
                        noise = newton_optimize(noise, grad, hessian, lr=LR)
                    else:
                        # 估计梯度
                        grad = estimate_gradient(model, vision_x + noise, inputs, target_output)
                        noise = noise - LR * grad

            attack_vision_x = vision_x + noise
            # 最终生成的输出
            generated_tokens = model.generate(
                vision_x=attack_vision_x.half().cuda(),
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

            # 每次迭代更新进度条
            progress_bar.update(1)



