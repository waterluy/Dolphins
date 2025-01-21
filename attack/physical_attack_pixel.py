import sys
sys.path.insert(0, '.')
import torch
from PIL import Image
import argparse
import os
from tqdm import tqdm
import cv2
from torchvision import transforms
import clip
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--iter', type=int, default=160)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--loss', type=str, default='cos', choices=['cos', 'kl'])
    parser.add_argument('--lamb1', type=float, default=0.75)
    parser.add_argument('--lamb2', type=float, default=0.75)
    parser.add_argument('--lamb3', type=float, default=0.05)
    args = parser.parse_args()
    return args

def text_supervision(
        ori_img,
        noise_start,
        text_features,
):
    noisy_img = ori_img.clone().cuda()
    noisy_img = noisy_img + noise_start.cuda()
    image_features = model_clip.encode_image(transform_clip(noisy_img))
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

def clean_supervision(
        ori_img,
        noise_start,
):
    noisy_img = ori_img.clone().cuda()
    noisy_img = noisy_img + noise_start.cuda()
    clean_features = model_clip.encode_image(transform_clip(ori_img.cuda()))
    noise_features = model_clip.encode_image(transform_clip(noisy_img.cuda()))
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
        ori_img,
        noise_start,
):
    noisy_img = ori_img.clone().cuda()
    noisy_img = noisy_img + noise_start.cuda()
    # 定义目标文本和其他文本
    texts = ["A safe driving scenario.", "An unsafe driving scenario."]
    text_tokens = clip.tokenize(texts).cuda()
    adv_logits_per_image, _ = model_clip(transform_clip(noisy_img.cuda()), text_tokens)
    adv_logits_per_image = torch.softmax(adv_logits_per_image, dim=-1)  # 1, 2
    clean_logits_per_image, _ = model_clip(transform_clip(ori_img.cuda()), text_tokens)
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

def coi_attack_stage2(
        induction_text,
        noise_start,
        ori_img,
        momentum
):    
    texts = [induction_text for _ in range(ori_img.shape[0])]
    alpha = 2 * EPS / ITER

    for _ in range(ITER):
        total_loss = 0
        noise_start.requires_grad = True
        text_features = model_clip.encode_text(clip.tokenize(texts).cuda())
        # print(text_features.shape)  # torch.Size([16, 512])

        loss_text = text_supervision(
            ori_img=ori_img,
            noise_start=noise_start,
            text_features=text_features,
        )
        total_loss = total_loss + LAMB1 * loss_text
        
        loss_clean = clean_supervision(
            ori_img=ori_img,
            noise_start=noise_start,
        )
        total_loss = total_loss + LAMB2 *  loss_clean

        loss_adj = adj_supervision(
            ori_img=ori_img,
            noise_start=noise_start,
        )
        total_loss = total_loss + LAMB3 * loss_adj

        noise_start.grad = None
        total_loss.backward()
        gradient = noise_start.grad
        gradient_l1 = torch.norm(gradient, 1)
        momentum = 1.0 * momentum + gradient / gradient_l1
        noise_start = noise_start.detach() - alpha * momentum.sign()
        noise_start = torch.clamp(noise_start, -EPS, EPS)

    return noise_start.detach(), momentum.detach()

def coi_attack_stage1(
        ori_img,
):
    noise = 2 * torch.rand_like(ori_img) - 1
    noise = noise * EPS
    noise.requires_grad = True
    momentum = 0
    induction_text = 'Turn left to avoid the obstacle!'
    
    for _ in range(args.query):
        noise, momentum = coi_attack_stage2(
            induction_text, 
            noise_start=noise,
            ori_img=ori_img,
            momentum=momentum,
        )
    return noise.detach()


if __name__ == '__main__':
    args = parse_args()
    EPS = args.eps
    ITER = args.iter
    QUERY = args.query
    LOSS = args.loss
    LAMB1 = args.lamb1
    LAMB2 = args.lamb2
    LAMB3 = args.lamb3
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model_clip, preprocess_clip = clip.load("ViT-B/32", device=torch.device('cuda')) 
    model_clip.eval()
    transform_clip = transforms.Compose([
        transforms.Resize(size=224, interpolation=3, antialias=True),  # bicubic插值方式（interpolation=3代表bicubic）
        transforms.CenterCrop(size=(224, 224)),  # 中心裁剪为224x224大小
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))  # 归一化
    ])

    # pics
    clean_folder = 'pic/clean'
    adv_folder = 'pic/adv'
    img_path_list = os.listdir(clean_folder)
    transform_totensor = transforms.ToTensor()
    for path in tqdm(img_path_list):
        frames = [Image.open(os.path.join(clean_folder, path)).convert('RGB')]
        images = torch.stack([transform_totensor(image) for image in frames], dim=0).to(device)
        # from torchvision.utils import save_image
        # save_image(images.squeeze()[0], "input.png")
        noise = coi_attack_stage1(images)
        final_inputs = images + noise.to(images.device, dtype=images.dtype)
        final_inputs = final_inputs.detach().squeeze()
        assert final_inputs.dim() == 3, f"Expected input tensor to have 3 dimensions but got {final_inputs.dim()}"
        save_image(final_inputs, os.path.join(adv_folder, path))
        # quit()
        