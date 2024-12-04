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



# TODO: Implement the RMSNorm class.
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor):


# TODO: Implement the Attention class.
class Attention(torch.nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # query, key, value layer [batch, number_heads, sequence_length, hidden_size_per_head]


        # attention scores and attention mask [batch, number_heads, sequence_length, sequence_length]
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))


        # context_layer [batch, sequence_length, hidden_size]
        return context_layer


# TODO: Implement the AttentionBlock class.
class AttentionBlock(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(AttentionBlock, self).__init__()

        self.dtype = dtype
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )


        self.core_attention = Attention(config)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):

        mixed_x_layer = self.query_key_value(hidden_states)

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

        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # TODO

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # TODO

        # =================
        # Output. [sequence_length, batch, hidden size]
        # =================
        return output, new_kv_cache


# TODO: Implement the MLP class.
class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(MLP, self).__init__()

    def forward(self, hidden_states):

        return output

# TODO: Implement the Layer class.
class Layer(torch.nn.Module):
    def __init__(self, config, device):
        super(Layer, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.self_attention = AttentionBlock(config, device=device, dtype=self.dtype)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [sequence_length, batch, hidden size]


        # output: [sequence_length, batch, hidden size]
        return output, new_kv_cache

# TODO: Implement the Transformer class.
class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm


    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):

        # new_kv_caches is a tuple
        # length: num_layers, each element is a tuple of length 2 (key, value cache)
        # key shape: [batch, multi_query_group_num, seq_len, kv_channels]
        return hidden_states, new_kv_caches


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
        pass
    torch.save(new_model_dict, "glm4.pt")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
    MODEL_PATH = "THUDM/glm-4-9b-chat"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    query = "你好"

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to(device)
    model = ChatGLMForConditionalGeneration(config=Config(), device=device).eval()

    convert_ckpt()

    model.load_state_dict(torch.load("glm4.pt"))

    generation_config = GenerationConfig(
        eos_token_id=[151329,151336,151338],
        pad_token_id= 151329,
        do_sample= True,
        temperature= 0.8,
        max_length= 100,
        top_p= 0.8,
        top_k= 1,
        transformers_version= "4.44.0")


    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))