
import json
import csv
import random
import sys
sys.path.append('.')

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

from configs.lora_config import openflamingo_tuning_config, otter_tuning_config

from mllm.src.factory import create_model_and_transforms

from huggingface_hub import hf_hub_download
from peft import (
    get_peft_model,
    LoraConfig,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel
)
from mllm.src.flamingo import ForwardType

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
        forward_type=ForwardType(FORWARDTYPE),
    )

    if CKPT is None:
        checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    else:
        checkpoint_path = CKPT
    print('load checkpoint from:', checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.half().cuda()

    return model, image_processor, tokenizer


def get_model_inputs(image_path, instruction, model, image_processor, tokenizer):
    frames = [Image.open(image_path)]
    vision_x = torch.stack([image_processor(image) for image in frames], dim=0).unsqueeze(0).unsqueeze(0)
    assert vision_x.shape[2] == len(frames)
    prompt = [
        f"USER: <image> is a driving video. {instruction} GPT:<answer>"
    ]
    inputs = tokenizer(prompt, return_tensors="pt", ).to(model.device)
   
    # print(vision_x.shape)   # torch.Size([1, 1, 16, 3, 336, 336])
    # print(prompt)

    return vision_x, inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, default='./playground/camouflage/clean')
    parser.add_argument('--output', type=str, default='./tmp')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--forward_type', type=int, default=0)
    args = parser.parse_args()
    CKPT = args.ckpt
    FORWARDTYPE = ForwardType(args.forward_type)
    model, image_processor, tokenizer = load_pretrained_modoel()
    generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                                'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                                'do_sample': False,
                                'early_stopping': True}

    with open('playground/camouflage/clean_perception.json', 'r') as file:
        data = json.load(file)

    folder = args.output
    json_path = os.path.join(folder, 'dolphin_output.json')
    os.makedirs(folder, exist_ok=True)
    
    instruction = "Please describe this image in detail."
    
    with open(json_path, 'w') as file:
        # 遍历JSON数据
        for index, (image_name, ground_truth) in tqdm(enumerate(data.items())):
            image_path = os.path.join(args.image_folder, image_name)

            tokenizer.eos_token_id = 50277
            tokenizer.pad_token_id = 50277

            try:
                vision_x, inputs = get_model_inputs(image_path, instruction, model, image_processor, tokenizer)
            except Exception as e:
                print(e)
                continue
            
            generated_tokens = model.generate(
                vision_x=vision_x.half().cuda(),
                lang_x=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                num_beams=3,
                forward_type=FORWARDTYPE,
                **generation_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            generated_text = tokenizer.batch_decode(generated_tokens)
            last_answer_index = generated_text[0].rfind("<answer>")
            content_after_last_answer = generated_text[0][last_answer_index + len("<answer>"):]
            final_answer = content_after_last_answer[:content_after_last_answer.rfind("<|endofchunk|>")]
            
            print('[Q]: ', instruction)
            print('[A]: ', final_answer)
            
            # 写入json行数据
            file.write(
                json.dumps({
                    "unique_id": image_name,
                    "pred": final_answer,
                    "gt": ground_truth,
                }) + "\n"
            )