import cv2
import os
import torch.nn.functional as F
import numpy as np
import torch
from math import sqrt

mean = [0.48145466, 0.4578275, 0.40821073] 
std = [0.26862954, 0.26130258, 0.27577711]

class GradCAM:
    def __init__(self, model, target_layer_name):
        """
        初始化 GradCAM 类，设置目标层并注册前向和反向钩子函数。
        
        参数：
            model: 模型
            target_layer_name: str, 目标层的名称, 指定在哪一层计算 Grad-CAM。
        """
        self.model = model
        self.target_layer = dict(self.model.named_modules())[target_layer_name]
        for name, param in self.target_layer.named_parameters():
            param.requires_grad = True
        self.activations = []
        self.gradients = []

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """
        存储特征图。
        
        参数：
            module: 当前层。
            input: 层的输入张量。
            output: torch.Tensor，层的输出张量，维度为 (channel, channels, height, width)。
        """
        # print(type(output)) # <class 'tuple'>
        # print(len(output))  # 2
        # print(type(output[0]))  # <class 'torch.Tensor'>
        # print(output[0].shape)  # torch.Size([577, 48, 1024])
        # print(type(output[1]))  # <class 'NoneType'>
        self.activations = [output[0]]

    def save_gradient(self, module, input, output):
        """
        存储反向传播中的梯度。
        
        参数：
            module: 当前层。
            input: 层的输入张量。
            output: torch.Tensor，层的输出梯度，维度为 (channel, channels, height, width)。
        """
        # print(type(output)) # <class 'tuple'>
        # print(len(output))  # 1
        # print(type(output[0]))  # <class 'torch.Tensor'>
        # print(output[0].shape)  # torch.Size([577, 48, 1024])
        self.gradients = [output[0]]

    def generate_cam(self, loss, retain_graph=False):
        """
        计算 Grad-CAM 可视化图。
        
        参数：
            loss: torch.Tensor，计算 Grad-CAM 所需的损失值。
            
        返回：
            torch.Tensor，Grad-CAM 图，维度为 (height, width)。
        """
        # 执行反向传播以获取梯度
        self.model.zero_grad()
        # print("\nbackward\n")
        loss.backward(retain_graph=retain_graph)
        # print("loss:")
        # print(loss)
        # print("\n")
        # 获取特征图和梯度
        if retain_graph:
            activation = self.activations[-1]
            gradients = self.gradients[-1]
        else:
            activation = self.activations[-1].detach()  # 获取保存的特征图
            gradients = self.gradients[-1].detach()   # 获取保存的梯度
        # print("activation:")
        # print(activation)    # activation.shape: torch.Size([577, 16, 1024])
        # print(activation.shape)
        # print("\n")
        gradients = torch.nan_to_num(gradients)
        # print("gradients:")
        # print(gradients)
        # print("\n")
        # print(gradients.shape)  # torch.Size([577, 16, 1024])
        weights = gradients.mean(dim=(2), keepdim=True)  # 对空间维度进行平均，得到权重
        # print(weights.shape)    # torch.Size([577, 16, 1])
        # print("weights:")
        # print(weights.squeeze())
        # print("\n")

        # 计算 Grad-CAM 权重和特征图的加权和
        cam = (weights * activation).sum(dim=0)  # 加权求和，合并【通道维度】
        # print("cam.shape")
        # print(cam.shape)    # torch.Size([16, 1024])
        # print("\n")
        cam = F.relu(cam)  # 应用 ReLU 以消除负值

        # 调整 CAM 尺寸以匹配图像
        # print("cam 0,1")
        # print(cam)
        # print(cam.shape)
        # tmp = cv2.applyColorMap((255 * cam.squeeze().cpu().numpy()).astype(np.uint8), cv2.COLORMAP_JET)
        # tmp = cv2.resize(tmp, (336, 336))
        # cv2.imwrite(os.path.join("grad_cam_images11", 'tmp.png'), tmp)
        # print("\n")
        side = int(sqrt(cam[0].size(0)))
        cam_frames = torch.stack([cam[i].view(side, side) for i in range(cam.size(0))])
        # print(cam_frames.shape)   # torch.Size([16, 32, 32])
        cam_frames = F.interpolate(cam_frames.unsqueeze(0), size=(336, 336), mode='bicubic', align_corners=False)
        # print("cam_frames interpolated:")
        # print(cam_frames)
        # print(cam_frames.shape)  # torch.Size([16, 336, 336])
        # print("\n")

        # 归一化到 [0, 1] 范围
        cam_frames  = cam_frames - cam_frames.min()
        cam_frames = cam_frames / cam_frames.max()
        cam_frames = cam_frames.squeeze()  # 去除多余的维度

        return cam_frames
    def save_cam_image(self, cam_frames, image_tensor, output_folder, image_name):
        """保存 Grad-CAM 热力图覆盖在原始图像上的结果。
        
        Args:
            cam (torch.Tensor): Grad-CAM 权重，形状为 (height, width)。
            image_tensor (torch.Tensor): 输入图像张量，形状为 (num, channels, height, width) 或 (channels, height, width)。
            output_path (str): 图像保存路径。
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # 如果只有一张图像，将其升维为 (1, channels, height, width)
        assert image_tensor.dim() == 4, "输入图像张量必须为 (num, channels, height, width) 或 (channels, height, width) 格式"

        cam_frames = cam_frames.squeeze().cpu().numpy()
        cam_frames = (cam_frames * 255).astype(np.uint8)

        os.makedirs(output_folder, exist_ok=True)
        image_mean = torch.tensor(mean).view(3, 1, 1).to(image_tensor.device)
        image_std = torch.tensor(std).view(3, 1, 1).to(image_tensor.device)
        # print("cam 0,255")
        # print(cam)
        # print("\n")
        # 对每张图像应用可视化
        for idx in range(image_tensor.shape[0]):
            heatmap = cv2.applyColorMap(cam_frames[idx], cv2.COLORMAP_JET)
            # print("heatmap:")
            # print(heatmap)
            # print("\n")
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 提取当前图像张量并转换为图像格式
            img_np = image_tensor[idx] * image_std + image_mean
            # img_np = img_np.cpu().numpy()
            img_np = img_np.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 合成热力图与原始图像
            superimposed_img = 0.80 * heatmap + img_np
            superimposed_img = superimposed_img / np.max(superimposed_img) * 255
            # superimposed_img = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            # 保存图像到本地
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_hm_bgr2rgb.png"), cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_hm_rgb2bgr.png"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_hm.png"), heatmap)
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_ori_rgb2bgr.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_ori_bgr2rgb.png"), cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}_ori.png"), img_np)
            cv2.imwrite(os.path.join(output_folder, f"{image_name}_{idx}.png"), superimposed_img)

            break
