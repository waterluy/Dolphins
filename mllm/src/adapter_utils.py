import torch.nn as nn
from transformers.activations import get_activation
from dataclasses import dataclass

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    # 适配器的输入维度
    d_model: int = 768
    # 激活函数
    non_linearity: str = "gelu_new"
    reduction_factor: int = 16



class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())  # 激活函数
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)  # 下采样
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)  # 上采样


    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output
    
class AdapterWithResidual(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())  # 激活函数
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)  # 下采样
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)  # 上采样

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        # 加残差
        output = output + x
        return output

class AdapterWithLayerNorm(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.d_model
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = Activations(config.non_linearity.lower())  # 激活函数
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)  # 下采样
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)  # 上采样
        self.layernorm = nn.LayerNorm(self.input_dim)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        output = self.layernorm(output)
        return output
    
    