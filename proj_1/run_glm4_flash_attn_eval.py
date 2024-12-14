import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import GenerationConfig
import torch.nn.functional as F
from typing import List
import time

from transformers.utils import is_flash_attn_2_available


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

@dataclass
class Config:
    hidden_size: int = 4096
    ffn_hidden_size: int = 13696
    kv_channels: int = 128
    num_layers: int = 40
    num_attention_heads: int = 32
    multi_query_attention: bool = True
    multi_query_group_num: int = 2
    padded_vocab_size: int = 151552
    seq_length: int = 8192
    layernorm_epsilon: float = 0.00000015625
    torch_dtype: float = "bfloat16"

    add_qkv_bias: bool = True # Means that the qkv linear layers have bias terms.
    post_layer_norm: bool = True # At the end of all layers, there is an additional RMSNorm.
    add_bias_linear: bool = False  # The linear layers in the FFN do not have bias terms.

    is_encoder_decoder: bool = False
    apply_residual_connection_post_layernorm: bool = False
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super(RMSNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        rms = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(rms + self.eps)
        return (self.gamma * hidden_states).to(input_dtype)


class Attention(torch.nn.Module):
    def __init__(self, config, layer_number):
        super(Attention, self).__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_causal = True
        # 计算投影尺寸
        projection_size = config.kv_channels * config.num_attention_heads

        # 每层的维度计算
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        coeff = self.layer_number
        self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # 计算注意力得分并进行缩放
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= self.norm_factor

        attention_scores = attention_scores.view(*output_size)

        attention_scores = attention_scores.float() * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        attention_probs = self.attention_dropout(attention_probs)

        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.view(*output_size)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class FlashAttention2(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = False

    def forward(self, query_states, key_states, value_states, attention_mask):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        batch_size, query_length = query_states.shape[:2]
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        dropout = self.config.attention_dropout if self.training else 0.0
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
            )
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size_per_partition).contiguous()
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads_per_partition, head_dim),
                indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class AttentionBlock(torch.nn.Module):

    def __init__(self, config, layer_number, device=None, dtype=torch.bfloat16):
        super(AttentionBlock, self).__init__()
        self.projection_size = config.kv_channels * config.num_attention_heads
        self.layer_number = layer_number
        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads # 每个头的键值维度
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        # 计算QKV隐藏层大小
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        # 定义qkv共用参数
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, dtype=dtype
                                         )

        # Attention算子
        self.core_attention = FlashAttention2(config, self.layer_number)
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, dtype=dtype
                               )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # 计算qkv投影，并分割为query, key, value
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # 位置编码
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)
        if use_cache:
            if kv_cache is None:
                kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                     dim=1)
            else:
                kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )
        # 合并多头输出，并通过最终线性层
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        output = self.dense(context_layer)

        return output, kv_cache


class MLP(torch.nn.Module):
    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            dtype=dtype
        )

        self.fc2 = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            dtype=dtype
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        x = self.fc1(hidden_states)
        x_g, x_l = x.chunk(2, dim=-1)
        x = F.silu(x_g) * x_l
        output = self.fc2(x)
        return output

class Layer(torch.nn.Module):
    def __init__(self, config, device, layer_number):
        super(Layer, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.self_attention = AttentionBlock(config, layer_number, device=device, dtype=self.dtype)
        self.mlp = MLP(config, device=device, dtype=self.dtype)

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=self.dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=self.dtype)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, new_kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache, use_cache
        )

        residual_connection = hidden_states + attention_output
        normed_residual_connection = self.post_attention_layernorm(residual_connection)

        ffn_output = self.mlp(normed_residual_connection)
        output = residual_connection + ffn_output

        return output, new_kv_cache


class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.layers = nn.ModuleList([Layer(config, device, _ + 1) for _ in range(self.num_layers)])

        if self.post_layer_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                device=device,
                dtype=self.dtype
            )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        # 存储每一层隐藏状态
        all_hidden_states = [] if output_hidden_states else None

        # 初始化kv cache
        new_kv_caches = [] if kv_caches is not None else None

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # 如果已有kv cache
            layer_kv_cache = kv_caches[layer_idx] if kv_caches is not None else None
            # 前向传播
            hidden_states, new_layer_kv_cache = layer(
                hidden_states, attention_mask, rotary_pos_emb,
                kv_cache=layer_kv_cache, use_cache=use_cache
            )
            # 保存新的kv cache
            if new_kv_caches is not None:
                new_kv_caches.append(new_layer_kv_cache)
        
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if new_kv_caches is not None:
            return hidden_states, tuple(new_kv_caches)
        else:
            return hidden_states, None


class GLM4(torch.nn.Module):
    def __init__(self, config, device):
        super(GLM4, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.word_embedding = torch.nn.Embedding(config.padded_vocab_size, config.hidden_size, dtype=self.dtype, device=device)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length

        self.model = Transformer(config, device=device)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False, dtype=self.dtype, device=device)
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=1,
                                              original_impl=True,
                                              device=device, dtype=self.dtype)

    def word_embedding_forward(self, input_ids):
        return self.word_embedding(input_ids)

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        if self.config._attn_implementation == "flash_attention_2":
            if padding_mask is not None and not padding_mask.all():
                return padding_mask
            return None
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def forward(self, input_ids, position_ids = None,
                past_key_values=None, full_attention_mask=None, attention_mask=None,
                use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embedding_forward(input_ids)
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        hidden_states, presents = self.model(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        return hidden_states, presents


class ChatGLMForConditionalGeneration(PreTrainedModel):
    def __init__(self, config, device=None):
        pretrain_config = PretrainedConfig(is_decoder=True, is_encoder_decoder=False)
        super().__init__(pretrain_config)

        self.max_sequence_length = 2500
        self.transformer = GLM4(config, device=device)
        self.config = config

    def _update_model_kwargs_for_generation(
            self,
            outputs,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=transformer_outputs[1],
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )


def convert_ckpt():
    huggingface_model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    model_dict = huggingface_model.state_dict()
    new_model_dict = {}
    for k, v in model_dict.items():
        if k == "transformer.embedding.word_embeddings.weight":
            new_model_dict["transformer.word_embedding.weight"] = v
        elif k == "transformer.rotary_pos_emb.inv_freq":
            new_model_dict["transformer.rotary_pos_emb.inv_freq"] = v
        elif "transformer.encoder.layers" in k:
            new_k = k.replace("transformer.encoder.layers", "transformer.model.layers")
            new_k = new_k.replace("mlp.dense_h_to_4h", "mlp.fc1")
            new_k = new_k.replace("mlp.dense_4h_to_h", "mlp.fc2")
            new_k = new_k.replace("input_layernorm.weight", "input_layernorm.gamma")
            new_k = new_k.replace("post_attention_layernorm.weight", "post_attention_layernorm.gamma")
            new_model_dict[new_k] = v
        elif k == "transformer.encoder.final_layernorm.weight":
            new_model_dict["transformer.model.final_layernorm.gamma"] = v
        else:
            new_model_dict[k] = v

    torch.save(new_model_dict, "glm4.pt")


if __name__ == "__main__":
    # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
    os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
    MODEL_PATH = "THUDM/glm-4-9b-chat"

    # 设置Cuda device并使用gpm4的tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    queries = [
        "你好",
        "你好", 
        "你好", 
        "你喜欢什么颜色？",
        "太阳从哪里升起？",
        "狗是如何表达情感的？",
        "你认为人工智能会对未来的工作市场产生什么影响？", 
        "简述一下中国的历史朝代以及它们的特点。", 
        "请介绍一下量子计算的基本原理。",
        "请详细解释一下人工智能的发展历史，包括重要的里程碑事件。",
        "描述一下地球生态系统的构成，涉及生物圈、气候变化和人类活动的影响。",
        "你如何看待全球气候变化问题？可以从政治、经济、社会和科技角度分析。",
        "请写一篇详细的文章，讨论可持续发展的多个方面，包括环境保护、经济增长、社会公平等，并给出具体案例。",
        "从科学、哲学、伦理和法律的角度分析人类基因编辑技术的潜力和挑战。",
        "以未来20年为时间框架，探讨人工智能如何影响全球经济、社会结构和文化变迁。"
    ]

    model = ChatGLMForConditionalGeneration(config=Config(), device=device).eval()

    model.load_state_dict(torch.load("glm4.pt", weights_only=True))

    generation_config = GenerationConfig(
        eos_token_id=[151329,151336,151338],
        pad_token_id=151329,
        do_sample=True,
        temperature=0.8,
        max_length=10000,
        top_p=0.8,
        top_k=1,
        transformers_version="4.44.0"
    )

    for query in queries:
        print(f"测试输入: {query}")
        
        # Tokenize input query
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors="pt",
                                               return_dict=True)
        inputs = inputs.to(device)

        input_token_count = inputs['input_ids'].shape[1]
        print(f"输入 token 数: {input_token_count}")

        # 记录显存和时间
        torch.cuda.empty_cache()  # 清理显存
        start_time = time.time()  # 开始计时
        initial_memory = torch.cuda.memory_allocated()  # 记录初始显存使用

        # 生成输出
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]  # 移除输入部分
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 获取输出的 token 数
        output_token_count = outputs.shape[1]
        print(f"输出 token 数: {output_token_count}")
        print("输出文本:", output_text)

        # 计算时间和显存
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算时间差
        final_memory = torch.cuda.memory_allocated()  # 获取最终显存使用量
        max_memory = torch.cuda.max_memory_allocated()  # 获取最大显存使用量

        print(f"运行时间: {elapsed_time:.4f} 秒")
        print(f"初始显存使用: {initial_memory / 1024 ** 2:.2f} MB")
        print(f"最终显存使用: {final_memory / 1024 ** 2:.2f} MB")
        print(f"最大显存使用: {max_memory / 1024 ** 2:.2f} MB\n")

    # 清理显存
    torch.cuda.empty_cache()