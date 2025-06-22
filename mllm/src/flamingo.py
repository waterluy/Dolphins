import torch
from einops import rearrange
from torch import nn
from .helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
import enum
from .utils import apply_with_stopping_condition
import torch.nn.functional as F
from .adapter_utils import AdapterConfig, Adapter, AdapterWithLayerNorm, AdapterWithResidual
  
class ForwardType(enum.Enum):
    Default = 0
    Imbeddings = 1
    Adapterwl0318 = 5
    Denoisewl0320 = 6
    AdapterWithResidual = 7
    Adapter2attack = 8
    AdapterNoShare = 105
    AdapterWithResidualNoShare = 107
    AdapterForVisual = 115
    AdapterWithResidualForVisual = 117
    AdapterKeyEntropyAtten = 11
    DefaultKeyEntropyAtten = 12
    AdapterBothKeyEntropyAtten = 13
    AdapterResKeyEntropyAtten = 14
    AdapterResBothKeyEntropyAtten = 15
    DefaultBothKeyEntropyAtten = 16
    AdapterNoShareBothKeyEntropyAtten = 17
    AdapterWithResidualNoShareBothKeyEntropyAtten = 18
    AdvPT = 19

class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        max_num_frames: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        forward_type=ForwardType.Default,
        lamb: float = 0.1,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.lamb = lamb
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.vision_encoder = vision_encoder.visual
        self.perceiver = PerceiverResampler(dim=self.vis_dim, max_num_frames=max_num_frames)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        # 加入adapter
        self.forward_type = forward_type
        if forward_type in [
            ForwardType.Adapterwl0318, ForwardType.Adapter2attack,
            ForwardType.AdapterForVisual, 
            ForwardType.AdapterKeyEntropyAtten,
            ForwardType.AdapterBothKeyEntropyAtten,
        ]:
            print("forward_type:", forward_type)
            self.at_adapter = Adapter(config=AdapterConfig(d_model=self.vis_dim))
        if forward_type == ForwardType.Denoisewl0320:
            print("forward_type:", forward_type)
            self.at_adapter = AdapterWithLayerNorm(config=AdapterConfig(d_model=self.vis_dim))
        if forward_type in [
            ForwardType.AdapterWithResidual, ForwardType.AdapterWithResidualForVisual,
            ForwardType.AdapterResKeyEntropyAtten, ForwardType.AdapterResBothKeyEntropyAtten,
        ]:
            print("forward_type:", forward_type)
            self.at_adapter = AdapterWithResidual(config=AdapterConfig(d_model=self.vis_dim))
        if forward_type in [
            ForwardType.AdapterNoShare,
            ForwardType.AdapterNoShareBothKeyEntropyAtten
        ]:
            # adapter 参数不共享
            print("forward_type:", forward_type)
            self.at_adapter = nn.ModuleList(
                [Adapter(config=AdapterConfig(d_model=self.vis_dim)) for _ in range(self.lang_encoder._get_decoder_layers())]
            )
        if forward_type in [
            ForwardType.AdapterWithResidualNoShare,
            ForwardType.AdapterWithResidualNoShareBothKeyEntropyAtten
        ]:
            # adapter 参数不共享
            print("forward_type:", forward_type)
            self.at_adapter = nn.ModuleList(
                [AdapterWithResidual(config=AdapterConfig(d_model=self.vis_dim)) for _ in range(len(self.lang_encoder._get_decoder_layers()))]
            )
        if forward_type in [
            ForwardType.AdvPT,
        ]:
            print("forward_type:", forward_type)
            # Prompt Tuning 参数
            self.prompt_length = 10  # 可调节的prompt token数量
            self.prompt_embed_dim = 4096  # 与模型隐藏层一致
            self.prompt_embeddings = nn.Parameter(
                torch.randn(self.prompt_length, self.prompt_embed_dim),
                requires_grad=True
            )
        
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing
        self.device = self.lang_encoder.device

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        media_locations : torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        forward_type=ForwardType.Default,
        adv_imgs=None,
        key_mode='normal',
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if forward_type in [
                ForwardType.Default,
                ForwardType.DefaultKeyEntropyAtten,  # ?????????
                ForwardType.DefaultBothKeyEntropyAtten,
                ForwardType.AdvPT,
            ]:
                self._encode_vision_x_original(vision_x=vision_x)
            elif forward_type in [
                ForwardType.Adapterwl0318, 
                ForwardType.AdapterWithResidual,
                ForwardType.AdapterNoShare,
                ForwardType.AdapterNoShareBothKeyEntropyAtten,
                ForwardType.AdapterWithResidualNoShare, 
                ForwardType.AdapterWithResidualNoShareBothKeyEntropyAtten, 
                ForwardType.AdapterKeyEntropyAtten, #主要在lang_encoder中修改
                ForwardType.AdapterBothKeyEntropyAtten,
                ForwardType.AdapterResKeyEntropyAtten,
                ForwardType.AdapterResBothKeyEntropyAtten,
                ForwardType.AdapterForVisual,
                ForwardType.AdapterWithResidualForVisual,
            ]:
                self._encode_vision_x_with_adapterwl0318(vision_x=vision_x, forward_type=forward_type)
            elif forward_type == ForwardType.Denoisewl0320:
                if adv_imgs is not None:
                    # 用于训练模型, 只denoise, 拿到denoise的损失
                    return self._encode_vision_x_only_denoisewl0320(vision_x=vision_x, adv_vision_x=adv_imgs)
                else:
                    # 用于攻击 走带有denoise的完整的流程
                    self._encode_vision_x_with_denoisewl0320(vision_x=vision_x)
            else:
                raise NotImplementedError(
                    f"forward_type {forward_type} is not implemented."
                )
            self._condition_media_locations(input_ids=lang_x)
        if forward_type in [
            ForwardType.AdapterKeyEntropyAtten,
            ForwardType.DefaultKeyEntropyAtten,
            ForwardType.AdapterBothKeyEntropyAtten,
            ForwardType.AdapterResKeyEntropyAtten,
            ForwardType.AdapterResBothKeyEntropyAtten,
            ForwardType.DefaultBothKeyEntropyAtten,
            ForwardType.AdapterNoShareBothKeyEntropyAtten,
            ForwardType.AdapterWithResidualNoShareBothKeyEntropyAtten,
        ]:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask,
                labels=labels,
                media_locations=media_locations,
                past_key_values=past_key_values,
                use_cache=use_cache,
                use_attn=True,
                lamb=self.lamb,
                key_mode=key_mode,
            )
        elif forward_type in [
            ForwardType.Default,
            ForwardType.Imbeddings,
            ForwardType.Adapterwl0318,
            ForwardType.Denoisewl0320,
            ForwardType.AdapterWithResidual,
            ForwardType.Adapter2attack,
            ForwardType.AdapterNoShare,
            ForwardType.AdapterWithResidualNoShare,
            ForwardType.AdapterForVisual,
            ForwardType.AdapterWithResidualForVisual,
        ]:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask,
                labels=labels,
                media_locations=media_locations,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        elif forward_type in [
            ForwardType.AdvPT,
        ]:
            output = self.lang_encoder(
                input_ids=lang_x,
                attention_mask=attention_mask,
                labels=labels,
                media_locations=media_locations,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prompt_embeddings=self.prompt_embeddings,
            )
        else:
            raise NotImplementedError(
                f"forward_type {forward_type} is not implemented."
            )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def forward_trades(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        media_locations : torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        trades_ret=None,
        clean_logits=None,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x_original(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)
        
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            media_locations=media_locations,
            past_key_values=past_key_values,
            use_cache=use_cache,
            trades_ret=trades_ret,
            clean_logits=clean_logits,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        media_locations: torch.Tensor = None,
        num_beams=1,
        min_new_tokens=None,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        repetition_penalty=1.0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
        forward_type=ForwardType.Default,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_vision_x = True
        if forward_type in [
            ForwardType.Default,
            ForwardType.DefaultKeyEntropyAtten,  # ?????????
            ForwardType.DefaultBothKeyEntropyAtten,
            ForwardType.AdvPT,
            ]:
            self._encode_vision_x_original(vision_x=vision_x)
        elif forward_type in [
            ForwardType.Adapterwl0318, 
            ForwardType.AdapterWithResidual,
            ForwardType.AdapterNoShare,
            ForwardType.AdapterWithResidualNoShare, 
            ForwardType.AdapterNoShareBothKeyEntropyAtten,
            ForwardType.AdapterWithResidualNoShareBothKeyEntropyAtten, 
            ForwardType.AdapterKeyEntropyAtten, #主要在lang_encoder中修改
            ForwardType.AdapterBothKeyEntropyAtten,
            ForwardType.AdapterResKeyEntropyAtten,
            ForwardType.AdapterResBothKeyEntropyAtten,
            ForwardType.AdapterForVisual,
            ForwardType.AdapterWithResidualForVisual,
        ]:
            self._encode_vision_x_with_adapterwl0318(vision_x=vision_x, forward_type=forward_type)
        else:
            raise NotImplementedError(
                f"forward_type {forward_type} is not implemented."
            )

        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            media_locations=media_locations,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x_original(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        
        # with torch.no_grad():
        vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
        
    def get_adapter_features(
        self,
        vision_x: torch.Tensor,
    ):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")

        # with torch.no_grad():
        vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
            
        # 参数共享
        # print(vision_x.shape)   # torch.Size([2, 5, 64, 1024])
        vision_x = self.at_adapter(vision_x)
        
        return vision_x
            
    def _encode_vision_x_with_adapterwl0318(self, vision_x: torch.Tensor, forward_type: ForwardType):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")

        # with torch.no_grad():
        vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        if forward_type in [
            ForwardType.AdapterForVisual,
            ForwardType.AdapterWithResidualForVisual,
        ]:
            # 视觉适配器
            vision_x = self.at_adapter(vision_x)
        vision_x = self.perceiver(vision_x)
            
        if forward_type in [
                ForwardType.Adapterwl0318,
                ForwardType.AdapterWithResidual,
                ForwardType.AdapterKeyEntropyAtten, #主要在lang_encoder中修改
                ForwardType.AdapterBothKeyEntropyAtten,
                ForwardType.AdapterResKeyEntropyAtten,
                ForwardType.AdapterResBothKeyEntropyAtten,
            ]:
                # 参数共享
                # print(vision_x.shape)   # torch.Size([2, 5, 64, 1024])
                vision_x = self.at_adapter(vision_x)

                for layer in self.lang_encoder._get_decoder_layers():
                    layer.condition_vis_x(vision_x)           
        elif forward_type in [
                ForwardType.AdapterNoShare,
                ForwardType.AdapterNoShareBothKeyEntropyAtten,
                ForwardType.AdapterWithResidualNoShare,
                ForwardType.AdapterWithResidualNoShareBothKeyEntropyAtten,
            ]:
                # 参数不共享
                adapter_index=0
                for layer in self.lang_encoder._get_decoder_layers():
                    vision_x = self.at_adapter[adapter_index](vision_x)
                    layer.condition_vis_x(vision_x)
                    adapter_index+=1
        elif forward_type in [
                    ForwardType.AdapterForVisual,
                    ForwardType.AdapterWithResidualForVisual,
                ]:
            for layer in self.lang_encoder._get_decoder_layers():
                    layer.condition_vis_x(vision_x)    
        else: 
                raise ValueError(f"Unknown forward_type: {self.forward_type}")
            
    def _encode_vision_x_with_denoisewl0320(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        
        vision_x = self.at_adapter(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        
    def _encode_vision_x_only_denoisewl0320(self, vision_x: torch.Tensor, adv_vision_x: torch.Tensor):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        
        assert adv_vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = adv_vision_x.shape[:3]
        # assert F == 1, "Only single frame supported"

        adv_vision_x = rearrange(adv_vision_x, "b T F c h w -> (b T F) c h w")
        
        with torch.no_grad():
            adv_vision_x = self.vision_encoder(adv_vision_x)[1]
        adv_vision_x = rearrange(adv_vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        adv_vision_x = self.perceiver(adv_vision_x)
        
        # denoise
        denoise_vision_x = self.at_adapter(adv_vision_x)
        # print(torch.sum(visual_query))
        # print(torch.sum(adv_visual_query))
        # 用干净图像的visual query和adv visual query做对比，得到loss
        # print(torch.nn.functional.cosine_similarity(denoise_visual_query, visual_query, dim=-1).shape)  # torch.Size([3, 10])
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(denoise_vision_x, vision_x, dim=-1).mean()
        # print(cosine_loss)
        mse_loss = torch.nn.functional.mse_loss(denoise_vision_x, vision_x)
        # print(mse_loss)
        loss = cosine_loss + 0.5 * mse_loss  # 余弦损失主导，MSE 作为辅助
        return [loss]
        

    def set_grad_adapter0318(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False       
        for param in self.perceiver.parameters():
            param.requires_grad = False
        for param in self.lang_encoder.parameters():
            param.requires_grad = False
        for param in self.at_adapter.parameters():
            param.requires_grad = True


    def set_grad_adat(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False     
        for param in self.perceiver.parameters():
            param.requires_grad = True
        for param in self.lang_encoder.parameters():
            param.requires_grad = False

    def set_grad_visualat(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = True      
        for param in self.perceiver.parameters():
            param.requires_grad = False
        for param in self.lang_encoder.parameters():
            param.requires_grad = False
            
    def set_promtp_tuning(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False      
        for param in self.perceiver.parameters():
            param.requires_grad = False
        for param in self.lang_encoder.parameters():
            param.requires_grad = False
        self.prompt_embeddings.requires_grad = True

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )
            self.lang_encoder.set_output_embeddings(
                wrap(wrap(self.lang_encoder.get_output_embeddings()))
            )
            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id
        
        if self.forward_type in [
            ForwardType.AdvPT,
        ]:
            media_locations = F.pad(
                media_locations, 
                (self.prompt_length, 0), 
                value=False
            )  # [B, prompt_length + T_txt]

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
