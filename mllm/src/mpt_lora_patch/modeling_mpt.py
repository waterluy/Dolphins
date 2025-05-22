"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .attention import attn_bias_shape, build_attn_bias
from .blocks import MPTBlock
from .norm import NORM_CLASS_REGISTRY
from .configuration_mpt import MPTConfig
from .adapt_tokenizer import AutoTokenizerForMOD, adapt_tokenizer_for_denoising
from .hf_prefixlm_converter import add_bidirectional_mask_if_missing, convert_hf_causal_lm_to_prefix_lm
from .meta_init_context import init_empty_weights
from .param_init_fns import MODEL_INIT_REGISTRY, generic_param_init_fn_
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    _no_split_modules = ["MPTBlock"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MPTModel):
            module.gradient_checkpointing = value

class MPTModel(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        config._validate_config()
        super().__init__(config)
        self.gradient_checkpointing = False
        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
        self.wte = nn.Embedding(config.vocab_size, config.d_model, device=config.init_device)
        if not self.alibi:
            self.wpe = nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList([MPTBlock(device=config.init_device, **config.to_dict()) for _ in range(config.n_layers)])
        self.norm_f = norm_class(config.d_model, device=config.init_device)
        if config.init_device != 'meta':
            self.apply(self.param_init_fn)
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(self.attn_impl, config.n_heads, config.max_seq_len, self.alibi, prefix_lm=self.prefix_lm, causal=self.is_causal, use_sequence_id=self.attn_uses_sequence_id)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    if config.verbose:
                        warnings.warn(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)
        if config.verbose and config.verbose > 2:
            print(self)
        if 'verbose' not in self.config.init_config:
            self.config.init_config['verbose'] = self.config.verbose
        if self.config.init_config['verbose'] > 1:
            init_fn_name = self.config.init_config['name']
            warnings.warn(f'Using {init_fn_name} initialization.')

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    @torch.no_grad()
    def _attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
                self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.n_heads, self.config.max_seq_len, causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (attn_bias, None)

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError('attn_bias does not match the expected shape. ' + f'The last two dimensions should both be {self.config.max_length} ' + f'but are {s_k} and {s_q}.')
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor] = None,
                needs_weights: Optional[bool]=False, prompt_embeddings: Optional[nn.Parameter]=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        # 检查输入唯一性
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        # 计算历史缓存长度
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        else:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # 获取词嵌入
        if inputs_embeds is None:
            tok_emb = self.wte(input_ids)
        else:
            tok_emb = inputs_embeds
        
        # 添加 prompt tuning
        if prompt_embeddings is not None:
            batch_size = tok_emb.size(0)
            prompts = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, prompt_len, hidden]
            tok_emb = torch.cat([prompts, tok_emb], dim=1)  # [batch, prompt_len+seq_len, hidden]
            # 调整attention_mask以包含prompt位置
            prompt_mask = torch.ones(
                (batch_size, prompts.size(1)), 
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            # 更新序列长度（影响后续位置编码）
            seq_length = seq_length + prompts.size(1)
            seq_length_with_past = seq_length + past_key_values_length

        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
            if prompt_embeddings is not None:
                prefix_mask = F.pad(prefix_mask, (prompts.size(1), 0), value=1)  # prompt位置设为1
        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')
        if output_attentions:
            raise NotImplementedError('output_attentions is not implemented yet for MPT')
        #if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0] and self.training:
        #    raise NotImplementedError('MPT does not support training with left padding.')
        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError('sequence_id is a requ                                                                             ired argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
        S = seq_length
        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
        
        # 进行位置编码
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
                past_position = past_key_values[0][0].size(1)
            if S + past_position > self.config.max_seq_len:
                raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
            pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            if attention_mask is not None and not self.training:
                pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        # dropout嵌入后处理
        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)
        # 注意力相关
        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=x.dtype, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]

        all_hidden_states = () if output_hidden_states else None
        for (b_idx, block) in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                (x, past_key_value) = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    past_key_value,
                    attn_bias,
                    attention_mask,
                    self.is_causal,
                )
            else:
                # 添加attn_weight
                (x, past_key_value, attn_weight) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal,
                                                         needs_weights=needs_weights)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
        x = self.norm_f(x)
        # 添加attn_weight
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=attn_weight)

    def param_init_fn(self, module):
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, MPTBlock)

class MPTForCausalLM(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = MPTModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.transformer.wte

    def set_output_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer
    # lables.shape: [B, seqlen]
    def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor] = None, 
                use_attn: Optional[bool] = False, prompt_embeddings: Optional[nn.Parameter] = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # input_ids.shape torch.Size([2, 182])
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, inputs_embeds=inputs_embeds,
                                   needs_weights=use_attn, prompt_embeddings=prompt_embeddings)
        logits = F.linear(outputs.last_hidden_state, self.transformer.wte.weight)   # logits: [B, seqlen, V]
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        key_loss = 0
        ce_loss = 0
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            # F.cross_entropy: ignore_index: int = -100,
            if prompt_embeddings is not None:
                logits = logits[:, prompt_embeddings.shape[0]:, :]
                logits = logits.contiguous() if not logits.is_contiguous() else logits
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        if use_attn:
            key_mask = self.get_key_mask(outputs.attentions, labels, key_mask_ratio=0.1)
            prob = torch.softmax(logits, dim=-1)  # [B, seqlen, V]
            key_loss = self.entropy_key_loss(prob, key_mask, weight=1.0)
            loss = ce_loss + key_loss
        else:
            loss = ce_loss
        return CausalLMOutputWithPast(loss=loss, key_loss=key_loss, ce_loss=ce_loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)
        
    def entropy_key_loss(self, probs, key_mask, weight=0.1):
        """
        关键词熵最小化损失（鼓励关键词位置的输出概率更尖锐）
        
        probs:     [B, T, V] softmax后的输出
        key_mask:  [B, T], 关键词位置为1,其他
        weight:    此损失项的缩放因子
        """
        B, T, V = probs.shape
        # print(probs.shape, key_mask.shape)  # torch.Size([4, 511, 32000]) torch.Size([4, 511])
        probs = probs.reshape(-1, V)               # [B*T, V]
        key_mask = key_mask.reshape(-1).bool()     # [B*T]
        # print(probs.shape, key_mask.shape)  # torch.Size([2044, 32000]) torch.Size([2044]) 

        if key_mask.sum() == 0:
            return probs.mean() * 0  # 避免没有关键词位置时计算出nan !!!!!!

        key_probs = probs[key_mask]                # [N, V]
        # print(key_probs.shape)  # torch.Size([126, 32000])
        entropy = - (key_probs * key_probs.log()).sum(dim=-1)  # 每个位置的熵 [N]
        # print(entropy.shape)    # torch.Size([126])
        mean_entropy = entropy.mean()
        # print(mean_entropy)  # tensor(9.0183, device='cuda:0', grad_fn=<MeanBackward0>)
        # print(mean_entropy * weight)    # tensor(0.9018, device='cuda:0', grad_fn=<MulBackward0>)

        return mean_entropy * weight

    def get_key_mask(self, atten, labels, key_mask_ratio=0.1):
        # atten: [B, num_heads, seqlen_q, seqlen_k]
        # labels: [B, seqlen]
        bs, seqlen = labels.shape
        valid_mask = labels != -100
        atten = atten.detach()
        atten = atten.mean(dim=1)  # [B, T, T] 平均掉head
        atten = atten.mean(dim=1)   # [B, T] 平均掉 query 维度，只留下每个 token 作为 key 被关注的程度
        # 每个位置的值就表示：这个 token 作为 key 时在全局 query 视角下被关注的平均程度；
        # top-k
        key_mask = torch.zeros_like(valid_mask, dtype=torch.bool)   # 初始化为 False
        for i in range(bs):
            # nonzero 返回的格式是 [row, col]，这里只需要 row 即可, row表示第几个有效token, col表示有效token的索引
            # nonzero 返回的tensor一定是2维的 [n, 1]
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(-1)   # 有效的 token 索引
            # print(valid_indices)
            # numel函数返回张量中元素的个数
            if valid_indices.numel() == 0:
                continue
            k = max(1, int(valid_indices.numel() * key_mask_ratio))  # 计算 top-k 的 k 值
            topk_scores = atten[i, valid_indices]  # dim= 只考虑有效的 token
            # print(topk_scores)  # torch.Size([n])
            topk_indices = topk_scores.topk(k, largest=True).indices  # 找到 top-k 的索引
            # print(topk_indices)  # torch.Size([k])
            selected_positions = valid_indices[topk_indices]    # 获取对应的位置
            # print(selected_positions.shape)
            # print(selected_positions)
            key_mask[i, selected_positions] = True  # 标记为 key 位置
            # quit()
        return key_mask

    def param_init_fn(self, module):
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](module=module, n_layers=self.config.n_layers, d_model=self.config.d_model, **self.config.init_config)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, MPTBlock)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not implemented for MPT yet')
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get('use_cache') == False:
                raise NotImplementedError('MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'prefix_mask': prefix_mask, 'sequence_id': sequence_id, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache', True), "media_locations": kwargs.get('media_locations')}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [tuple((past_state.index_select(0, beam_idx) for past_state in layer_past))]
        return reordered_past