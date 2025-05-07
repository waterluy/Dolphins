import torch


class ATConfig:
    def __init__(self, args):
        self.at_iter = 10
        self.at_eps_imgs = 0.1
        self.at_eps_embd = 0.02
        self.at_prob = 0.6
        self._apply_arguments(args)
        self.at_alpha_imgs = 2 * self.at_eps_imgs / self.at_iter
        self.at_alpha_embd = 2 * self.at_eps_embd / self.at_iter

    def _apply_arguments(self, args):
        """仅将 at_ 开头的参数存储为类属性"""
        for key, value in vars(args).items():
            if key.startswith("at_") and hasattr(self, key):
                setattr(self, key, value)
    
    
class AdversarialNoise:
    def __init__(self, at_config: ATConfig):
        self.at_config = at_config
        self.image_noise = None
        self.embeddings_noise = None

    def generate_image_noise(self, imgs):
        """ 生成图像对抗噪声 """
        self.image_noise = 2 * torch.randn_like(imgs) - 1  # 初始化噪声
        self.image_noise = torch.clamp(self.image_noise, -self.at_config.at_eps_imgs, self.at_config.at_eps_imgs)
        self.image_noise.requires_grad = True
        return self.image_noise

    def update_image_noise(self, grad):
        """ 更新图像对抗噪声 """
        self.image_noise = self.image_noise.detach() + self.at_config.at_alpha_imgs * grad.sign()
        self.image_noise = torch.clamp(self.image_noise, -self.at_config.at_eps_imgs, self.at_config.at_eps_imgs)
        self.image_noise.requires_grad = True
        return self.image_noise

    def generate_text_noise(self, embeddings):
        """ 生成文本对抗噪声 """
        self.embeddings_noise = 2 * torch.randn_like(embeddings) - 1
        self.embeddings_noise = self.embeddings_noise * self.at_config.at_eps_embd
        self.embeddings_noise.requires_grad = True
        return self.embeddings_noise

    def update_text_noise(self, grad):
        """ 更新文本对抗噪声 """
        self.embeddings_noise = self.embeddings_noise.detach() + self.at_config.at_alpha_embd * grad.sign()
        self.embeddings_noise.requires_grad = True
        return self.embeddings_noise