import os
import sys
sys.path.append('.')
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Union
from PIL import Image
import mimetypes
import copy
import csv
import random
from tools.run_tools import dump_args
import cv2
import requests
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import LlamaTokenizer, CLIPImageProcessor
import pandas as pd
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
import clip
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import InterpolationMode

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
    return instructions

def normalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).half().to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).half().to(tensor.device)
    return (tensor - mean) / std

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).half().to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).half().to(tensor.device)
    return tensor * std + mean

def get_ad_3p(task):
    if task == "detailed_description":
        return "perception"
    elif task == "open_voc_object":
        return "perception"
    elif task == "scene":
        return "perception"
    elif task == "timeofday":
        return "perception"
    elif task == "traffic_light":
        return "perception"
    elif task == "weather":
        return "perception"
    else:
        raise ValueError("Invalid task name: {}".format(task))

def text_supervision(
        ori_vision_x,
        patch_start,
        text_features,
):
    resize_to_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None)
    denormed_vision_x = denormalize(ori_vision_x, mean=image_mean, std=image_std)[0, 0, :].cuda()
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = denormed_vision_x.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    denormed_vision_x = denormed_vision_x * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
    normed_noisy_vision_x = normalize(denormed_vision_x, mean=image_mean, std=image_std)
    image_features = model_clip.encode_image(resize_to_224(normed_noisy_vision_x))
    if LOSS == 'cos':
        text_features_normed = F.normalize(text_features, dim=-1)
        # print(text_features_normed.shape)   # torch.Size([16, 512])
        image_features_normed = F.normalize(image_features, dim=-1)
        # print(image_features_normed.shape)  # torch.Size([16, 512])
        total_loss = - torch.cosine_similarity(image_features_normed, text_features_normed, dim=1, eps=1e-8)
        # print(total_loss.shape) # torch.Size([16])
        total_loss = total_loss.mean()
        # print(total_loss, total_loss.shape) 
    elif LOSS == 'kl':
        # 将两个嵌入特征转换为概率分布, text的特征指导image的特征
        text_prob = F.softmax(text_features, dim=-1)       # 文本特征的概率分布
        image_log_prob = F.log_softmax(image_features, dim=-1)  # 图像特征的对数概率分布
        # 计算 KL 散度
        kl_divergence = F.kl_div(image_log_prob, text_prob, reduction='none')
        # 对 dim 维度求和，得到每个样本的 KL 散度，形状为 [batch_size]
        kl_divergence_per_sample = kl_divergence.sum(dim=-1)
        total_loss = kl_divergence_per_sample.mean()
        # KL 散度越大 表示两个分布的差异越大
    else:
        raise ValueError("Invalid loss type: {}".format(LOSS))
    return total_loss

def inverse3p_supervision(
        ori_vision_x,
        patch_start,
        ad_3p_stage,
):
    resize_to_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None)
    denormed_vision_x = denormalize(ori_vision_x, mean=image_mean, std=image_std)[0, 0, :].cuda()
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = denormed_vision_x.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    denormed_vision_x = denormed_vision_x * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
    normed_noisy_vision_x = normalize(denormed_vision_x, mean=image_mean, std=image_std)
    resize_to_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None)
    # 定义目标文本和其他文本
    texts = ["perception prediction plan", "perception prediction", "perception"]
    text_tokens = clip.tokenize(texts).cuda()
    logits_per_image, _ = model_clip(resize_to_224(normed_noisy_vision_x), text_tokens)
    logits_per_image = torch.softmax(logits_per_image, dim=-1)
    target_labels = torch.full(logits_per_image.shape, -1).cuda()   # 初始值为-1以抑制非目标类别
    if ad_3p_stage == 'perception':
        target_labels[:, 2] = 1
    elif ad_3p_stage == 'prediction':
        target_labels[:, 1] = 1
    elif ad_3p_stage == 'plan':
        target_labels[:, 0] = 1
    mask = target_labels != -1
    # 最大化target label 同时抑制其他label
    bs = logits_per_image.shape[0]
    loss = -torch.log(1e-8 + logits_per_image[mask].view(bs, -1)).mean(dim=-1, keepdim=True) + torch.log(1e-8 + logits_per_image[~mask].view(bs, -1)).mean(dim=-1, keepdim=True)
    loss = loss.mean(dim=0)
    return loss

def clean_supervision(
        ori_vision_x,
        patch_start,
):
    resize_to_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None)
    denormed_vision_x = denormalize(ori_vision_x, mean=image_mean, std=image_std)[0, 0, :].cuda()
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = denormed_vision_x.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    denormed_vision_x = denormed_vision_x * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
    normed_noisy_vision_x = normalize(denormed_vision_x, mean=image_mean, std=image_std)
    noise_features = model_clip.encode_image(resize_to_224(normed_noisy_vision_x))
            
    clean_features = model_clip.encode_image(resize_to_224(ori_vision_x[0, 0, :]).cuda())
    if LOSS == 'cos':
        clean_features_normed = F.normalize(clean_features, dim=-1)
        # print(text_features_normed.shape)   # torch.Size([16, 512])
        noise_features_normed = F.normalize(noise_features, dim=-1)
        # print(image_features_normed.shape)  # torch.Size([16, 512])
        total_loss = torch.cosine_similarity(clean_features_normed, noise_features_normed, dim=1, eps=1e-8)
        # print(total_loss.shape) # torch.Size([16])
        total_loss = total_loss.mean()
        # print(total_loss, total_loss.shape) 
    elif LOSS == 'kl':
        # 将两个嵌入特征转换为概率分布, clean的特征指导noise的特征
        clean_prob = F.softmax(clean_features, dim=-1)       # 文本特征的概率分布
        noise_log_prob = F.log_softmax(noise_features, dim=-1)  # 图像特征的对数概率分布
        # 计算 KL 散度
        kl_divergence = F.kl_div(noise_log_prob, clean_prob, reduction='none')
        # 对 dim 维度求和，得到每个样本的 KL 散度，形状为 [batch_size]
        kl_divergence_per_sample = kl_divergence.sum(dim=-1)
        total_loss = - kl_divergence_per_sample.mean()
        # KL 散度越大 表示两个分布的差异越大
    else:
        raise ValueError("Invalid loss type: {}".format(LOSS))
    return total_loss

def adj_supervision(
        ori_vision_x,
        patch_start,
):
    denormed_vision_x = denormalize(ori_vision_x, mean=image_mean, std=image_std)[0, 0, :].cuda()
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = denormed_vision_x.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    denormed_vision_x = denormed_vision_x * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
    normed_noisy_vision_x = normalize(denormed_vision_x, mean=image_mean, std=image_std)
    resize_to_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None)
    # 定义目标文本和其他文本
    texts = ["A safe driving scenario.", "A dangerous driving scenario."]
    text_tokens = clip.tokenize(texts).cuda()
    adv_logits_per_image, _ = model_clip(resize_to_224(normed_noisy_vision_x), text_tokens)
    adv_logits_per_image = torch.softmax(adv_logits_per_image, dim=-1)  # 1, 2
    clean_logits_per_image, _ = model_clip(resize_to_224(ori_vision_x[0, 0, :]).cuda(), text_tokens)
    clean_logits_per_image = torch.softmax(clean_logits_per_image, dim=-1)  # 1, 2

    target_labels = torch.full(adv_logits_per_image.shape, -1).cuda()   # 初始值为-1以抑制非目标类别
    # 找到 clean_logits_per_image 中较小元素的索引
    min_index = torch.argmin(clean_logits_per_image, dim=-1)  # 返回形状 [1] 的张量，表示较小元素的位置
    # 将 target_labels 中较小元素的位置设为 1
    target_labels[torch.arange(target_labels.shape[0]), min_index] = 1
    mask = target_labels == 1
    # 最大化target label 同时抑制其他label
    bs = adv_logits_per_image.shape[0]
    loss = -torch.log(1e-8 + adv_logits_per_image[mask].view(bs, -1)).mean(dim=-1, keepdim=True) + torch.log(1e-8 + adv_logits_per_image[~mask].view(bs, -1)).mean(dim=-1, keepdim=True)
    loss = loss.mean(dim=0)
    return loss

def apply_transform_and_generate_mask(patch, target_size):
    """应用随机仿射变换并生成掩码，放置在图像上方 30% 处的中心位置。
    参数：
        - patch: 原始补丁 (C, H, W)
        - target_size: 目标图像大小 (H', W')
    返回：
        - transformed_patch: 变换后的补丁 (C, H', W')
        - transformed_mask: 与补丁对应的掩码 (1, H', W')
    """
    bs, c, h, w = patch.shape
    
    random_affine = transforms.RandomAffine(
        degrees=(-5, 5), 
        translate=(0.05, 0.1), 
        scale=(0.90, 1.11), 
        shear=(0.1)
    )
    transform_with_probability = transforms.RandomApply(
        [random_affine],    # 要应用的变换列表
        p=0.1               # 应用变换的概率
    )

    # 计算填充以将补丁放置在目标位置
    center_x = int(pos_y * target_size[1]) - w // 2
    center_y = int(pos_y * target_size[0]) - h // 2
    
    # 填充补丁到目标图像大小
    padded_patch = F.pad(patch, (
        center_x, target_size[1] - w - center_x,
        center_y, target_size[0] - h - center_y
    ), mode='constant', value=0)
    
    # 应用随机仿射变换
    transformed_patch = transform_with_probability(padded_patch)
    
    # 生成掩码，仿射变换后的补丁非零区域为 1
    mask = (transformed_patch != 0).float().sum(dim=0, keepdim=True)
    transformed_mask = torch.clamp(mask, 0, 1)  # 转换为二值掩码 (1, H', W')
    
    return transformed_patch, transformed_mask

def coi_attack_stage2(
        induction_text,
        patch_start,
        optimizer,
        ori_vision_x,
):    
    texts = [induction_text for _ in range(vision_x.shape[2])]
    for _ in range(ITER):
        total_loss = 0
        patch_start.requires_grad = True
        text_features = model_clip.encode_text(clip.tokenize(texts).cuda())
        # print(text_features.shape)  # torch.Size([16, 512])
        if args.sup_text:
            loss_text = text_supervision(
            ori_vision_x=ori_vision_x,
            patch_start=patch_start,
            text_features=text_features,
        )
            # print(loss_text)
            total_loss = total_loss + loss_text
        if args.sup_clean:
            loss_clean = clean_supervision(
                ori_vision_x=ori_vision_x,
                patch_start=patch_start,
            )
            # print(0.02 * loss_clean)
            total_loss = total_loss + loss_clean
        if args.sup_adj:
            loss_adj = adj_supervision(
                ori_vision_x=ori_vision_x,
                patch_start=patch_start,
            )
            # print(0.02 * loss_clean)
            total_loss = total_loss + 0.05 * loss_adj
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        patch_start = torch.clamp(patch_start.detach(), 0, 1)

    return patch_start.detach()

def coi_attack_stage1(
        ori_vision_x,
        ori_inputs,
        texts,
):
    batch_size, c, h, w = ori_vision_x.shape[2:]  # 获取输入图像的高和宽
    patch_size = (int(h * patch_ratio), int(w * patch_ratio))  # 计算补丁大小
    input_image_size = ori_vision_x.shape[4:]  # 输入图像目标尺寸
    # 初始化对抗补丁
    adversarial_patch = torch.rand((batch_size, c, *patch_size), requires_grad=True)
    answers = []
    
    alpha = 2 * EPS / ITER
    optimizer = torch.optim.Adam([adversarial_patch], lr=alpha)
    for induction_text in texts:
        adversarial_patch = coi_attack_stage2(
            induction_text, 
            patch_start=adversarial_patch,
            optimizer=optimizer,
            ori_vision_x=ori_vision_x,
        )
        final_input_vision_x = ori_vision_x.clone()
        final_input_vision_x = denormalize(final_input_vision_x, mean=image_mean, std=image_std)
        transformed_patch, transformed_mask = apply_transform_and_generate_mask(adversarial_patch, input_image_size)
        final_input_vision_x[0, 0, :] = final_input_vision_x[0, 0, :] * (1 - transformed_mask) + transformed_patch * transformed_mask
        final_input_vision_x = normalize(final_input_vision_x, mean=image_mean, std=image_std)
        final_answer = inference(
            input_vision_x=final_input_vision_x.half().cuda(),
            inputs=ori_inputs
        )
        answers.append(final_answer)
        if final_answer == "":
            break
    return adversarial_patch.detach(), answers


def inference(input_vision_x, inputs):
    inference_tokens = model.generate(
        vision_x=input_vision_x.half().cuda(),
        lang_x=inputs["input_ids"].clone().cuda(),
        attention_mask=inputs["attention_mask"].clone().cuda(),
        num_beams=3,
        **generation_kwargs,
    )

    inference_tokens = inference_tokens.cpu().numpy()
    if isinstance(inference_tokens, tuple):
        inference_tokens = inference_tokens[0]

    inference_text = tokenizer.batch_decode(inference_tokens)
    last_answer_index = inference_text[0].rfind("<answer>")
    content_after_last_answer = inference_text[0][last_answer_index + len("<answer>"):]
    final_answer = content_after_last_answer[:content_after_last_answer.rfind("<|endofchunk|>")]
    return final_answer


image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT Evaluation')
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--query', type=int, default=8)
    parser.add_argument('--loss', type=str, default='cos', choices=['cos', 'kl'])
    parser.add_argument('--sup-clean', action='store_true')
    parser.add_argument('--sup-text', action='store_true')
    parser.add_argument('--sup-3p', action='store_true')
    parser.add_argument('--sup-adj', action='store_true')
    args = parser.parse_args()
    EPS = args.eps
    ITER = args.iter
    QUERY = args.query
    LOSS = args.loss
    # patch 超参数
    patch_ratio = 0.17  # 补丁相对图像大小的比例
    pos_x = 0.5
    pos_y = 0.3
    
    best_records_path = 'results/bench_attack_coi-opti_eps0.2_iter20_query8/records.json'
    best_records = []
    with open(best_records_path, 'r') as file:
        best_records = json.load(file)

    ok_unique_id = []
    iii = ''
    if args.sup_text:
        iii += '-text'
    if args.sup_3p:
        iii += '-3p'
    if args.sup_clean:
        iii += '-clean'
    if args.sup_adj:
        iii += '-adj'
    folder = f'results/bench_attack_coi-opti-judge-offline-{LOSS}-patchall-i2{iii}_eps{EPS}_iter{ITER}_query{QUERY}'
    os.makedirs(folder, exist_ok=True)
    dump_args(folder=folder, args=args)
    json_path = os.path.join(folder, 'dolphin_output.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            for line in file:
                ok_unique_id.append(json.loads(line)['unique_id'])

    induction_records = []
    coi_records_file = os.path.join(folder, 'records.json')
    
    model, image_processor, tokenizer = load_pretrained_modoel()
    tokenizer.eos_token_id = 50277
    tokenizer.pad_token_id = 50277
    device = model.device
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=torch.device('cuda')) 
    model_clip.eval()

    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}

    with open('playground/dolphins_bench/dolphins_benchmark.json', 'r') as file:
        data = json.load(file)

    try:
        with open(json_path, 'a') as file:
            # 遍历JSON数据
            for entry in tqdm(data):
                unique_id = entry["id"]
                label = entry['label']
                task_name = entry['task_name']
                video_path = entry['video_path'][entry['video_path'].find('/')+1:]
                # 从conversations中提取human的value和gpt的value
                instruction = entry['conversations'][0]['value']
                ground_truth = entry['conversations'][1]['value']

                if unique_id in ok_unique_id:
                    continue
                
                vision_x, inputs = get_model_inputs(video_path=video_path, instruction=instruction, model=model, image_processor=image_processor, tokenizer=tokenizer)
                
                now_dict = list(filter(lambda x: x["unique_id"] == unique_id, best_records))
                assert len(now_dict) == 1
                induction_texts = now_dict[0]["induction_records"]
                if QUERY < len(induction_texts):
                    induction_texts = induction_texts[:QUERY]
                
                adversarial_patch, induction_answers = coi_attack_stage1(ori_vision_x=vision_x, ori_inputs=inputs, texts=induction_texts)

                final_answer = induction_answers[-1]

                file.write(
                    json.dumps({
                        "unique_id": unique_id,
                        "task_name": task_name,
                        "pred": final_answer,
                        "gt": ground_truth,
                        "label": label
                    }) + "\n"
                )
                # 记录induction texts
                induction_records.append({
                    "unique_id": unique_id,
                    "task_name": task_name,
                    "induction_records": induction_texts,
                    "induction_answers": induction_answers,
                })
    finally:
        with open(coi_records_file, 'w') as file:
            json.dump(induction_records, file, indent=4)
