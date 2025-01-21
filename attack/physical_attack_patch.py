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
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--loss', type=str, default='cos', choices=['cos', 'kl'])
    parser.add_argument('--lamb1', type=float, default=0.75)
    parser.add_argument('--lamb2', type=float, default=0.75)
    parser.add_argument('--lamb3', type=float, default=0.05)
    args = parser.parse_args()
    return args

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


def text_supervision(
        ori_img,
        patch_start,
        text_features,
):
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = ori_img.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    noisy_img = ori_img * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
    
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
        patch_start,
):
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = ori_img.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    noisy_img = ori_img * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
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
        patch_start,
):
    # 生成与图像同尺寸的变换补丁和掩码
    input_image_size = ori_img.shape[2:]  # 输入图像目标尺寸
    transformed_patch, transformed_mask = apply_transform_and_generate_mask(patch_start, input_image_size)
    # 将补丁放置到图像的指定位置，仅覆盖非空白部分
    noisy_img = ori_img * (1 - transformed_mask.cuda()) + transformed_patch.cuda() * transformed_mask.cuda()
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
        patch_start,
        ori_img,
        optimizer,
):    
    texts = [induction_text for _ in range(ori_img.shape[0])]

    for _ in range(ITER):
        total_loss = 0
        patch_start.requires_grad = True
        text_features = model_clip.encode_text(clip.tokenize(texts).cuda())
        # print(text_features.shape)  # torch.Size([16, 512])

        loss_text = text_supervision(
            ori_img=ori_img,
            patch_start=patch_start,
            text_features=text_features,
        )
        total_loss = total_loss + LAMB1 * loss_text
        
        loss_clean = clean_supervision(
            ori_img=ori_img,
            patch_start=patch_start,
        )
        total_loss = total_loss + LAMB2 *  loss_clean

        loss_adj = adj_supervision(
            ori_img=ori_img,
            patch_start=patch_start,
        )
        total_loss = total_loss + LAMB3 * loss_adj

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        patch_start = torch.clamp(patch_start.detach(), 0, 1)

    return patch_start.detach()

def coi_attack_stage1(
        ori_img,
):
    batch_size, c, h, w = ori_img.shape  # 获取输入图像的高和宽
    patch_size = (int(h * patch_ratio), int(w * patch_ratio))  # 计算补丁大小
    # 初始化通用对抗补丁
    adversarial_patch = torch.rand((batch_size, c, *patch_size), requires_grad=True)
    alpha = 2 * EPS / ITER
    optimizer = torch.optim.Adam([adversarial_patch], lr=alpha)
    
    induction_text = 'Keep going!'
    
    for _ in range(args.query):
        adversarial_patch = coi_attack_stage2(
            induction_text, 
            patch_start=adversarial_patch,
            ori_img=ori_img,
            optimizer=optimizer,
        )
    return adversarial_patch.detach()


if __name__ == '__main__':
    args = parse_args()
    EPS = args.eps
    ITER = args.iter
    QUERY = args.query
    LOSS = args.loss
    LAMB1 = args.lamb1
    LAMB2 = args.lamb2
    LAMB3 = args.lamb3
    # patch 超参数
    ratios = [0.03, 0.06, 0.09, 0.12]
    patch_ratio = 0.17  # 补丁相对图像大小的比例
    pos_x = 0.5
    pos_y = 0.3
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
    signs_folder = 'signs'
    img_path_list = list(filter(lambda i: i.endswith('.png'), os.listdir(signs_folder)))
    transform_totensor = transforms.ToTensor()
    for path in tqdm(img_path_list):
        frames = [Image.open(os.path.join(signs_folder, path)).convert('RGB')]
        images = torch.stack([transform_totensor(image) for image in frames], dim=0).to(device)
        # from torchvision.utils import save_image
        # save_image(images.squeeze()[0], "input.png")
        patch = coi_attack_stage1(images)
        save_image(patch[0], os.path.join(signs_folder, 'patch', path.replace('.png', '_patch.png')))
        # quit()
        