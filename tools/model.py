import torch
import torch.nn as nn


class TextToNoiseGenerator(nn.Module):
    def __init__(self, text_dim, output_shape, eps):
        """
        Args:
            text_dim (int): 输入文本特征的维度
            output_shape (tuple): 输出噪声的形状 (c, h, w)
        """
        super(TextToNoiseGenerator, self).__init__()
        self.output_shape = output_shape
        c, h, w = output_shape
        self.eps = eps

        # 全连接层将文本特征映射到 c * h * w 的向量
        self.fc = nn.Sequential(
            nn.Linear(text_dim, c * h * w),
            nn.Tanh()  # 将输出限制在 [-1, 1] 范围内
        )


    def forward(self, text_features):
        """
        Args:
            text_features (tensor): 文本特征张量，形状为 [batch size, text_dim]
        
        Returns:
            noise (tensor): 输出噪声图像，形状为 [batch size, c, h, w]
        """
        batch_size = text_features.size(0)

        # 通过全连接层映射到 c * h * w
        x = self.fc(text_features)  # [batch size, c * h * w]

        # reshape 成 [batch size, c, h, w]
        noise = x.view(batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2])

        # 将输出缩放到 [-eps, eps] 范围内
        noise = noise * self.eps

        return noise

class GeneratorFromText(nn.Module):
    def __init__(self, text_dim, output_shape, num_filters, eps):
        """
        Args:
            text_dim (int): 输入文本特征的维度
            output_shape (tuple): 输出噪声的形状 (c, h, w)
            num_filters (list): 各个卷积层的通道数列表
            eps (float): 噪声的幅度限制
        """
        super(GeneratorFromText, self).__init__()
        self.eps = eps
        self.output_shape = output_shape
        c, h, w = output_shape
        self.init_dim = num_filters[0] * (h // 4) * (w // 4)
        self.num_filters = num_filters

        # 初始全连接层：将文本特征映射为图像特征格式
        self.fc = nn.Sequential(
            nn.Linear(text_dim, self.init_dim),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[2], c, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 将输出限制在 [-1, 1] 范围内
        )

    def forward(self, text_features):
        """
        Args:
            text_features (tensor): 文本特征张量，形状为 [batch size, text_dim]
        
        Returns:
            noise (tensor): 输出噪声图像，形状为 [batch size, c, h, w]
        """
        batch_size = text_features.size(0)

        # 通过全连接层映射到初始图像特征维度
        x = self.fc(text_features)  # [batch size, init_dim]
        x = x.view(batch_size, self.num_filters[0], self.output_shape[1] // 4, self.output_shape[2] // 4)

        # 通过卷积层逐步上采样到目标尺寸
        x = self.conv_layers(x)

        # 将输出缩放到 [-eps, eps] 范围内
        noise = x * self.eps

        return noise
