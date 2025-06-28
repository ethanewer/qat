from typing import Callable, Optional

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .quantization_util import LsqBinaryTernaryExtension, StretchedElasticQuant


class QATQwen3Attention(Qwen3Attention):
    def __init__(
        self,
        *args,
        w_bits: int = 16,
    ) -> None:
        super().__init__(*args)
        self.w_bits = w_bits
        if self.w_bits < 16:
            self.key_clip_val = nn.Parameter(torch.Tensor(self.k_proj.out_features))
            self.value_clip_val = nn.Parameter(torch.Tensor(self.v_proj.out_features))

    def forward(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: None = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        real_key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        real_value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.w_bits >= 16:
            key_states = real_key_states
            value_states = real_value_states
        elif self.w_bits == 2 or self.w_bits == 0:
            key_states = StretchedElasticQuant.apply(
                real_key_states,
                self.key_clip_val,
                self.w_bits,
                False,
            ).to(real_key_states.dtype)  # type: ignore
            value_states = StretchedElasticQuant.apply(
                real_value_states,
                self.value_clip_val,
                self.w_bits,
                False,
            ).to(real_value_states.dtype)  # type: ignore
        elif self.w_bits <= 4:
            key_states = LsqBinaryTernaryExtension.apply(
                real_key_states,
                self.key_clip_val,
                self.w_bits,
                False,
            ).to(real_key_states.dtype)  # type: ignore
            value_states = LsqBinaryTernaryExtension.apply(
                real_value_states,
                self.value_clip_val,
                self.w_bits,
                False,
            ).to(real_value_states.dtype)  # type: ignore
        else:
            raise NotImplementedError

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            raise NotImplementedError("Cache is not supported during QAT.")

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(  # type: ignore
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def patch_attention_with_qat_attention(self_attn: Qwen3Attention, w_bits: int = 16) -> None:
    self_attn.w_bits = w_bits  # type: ignore
    if w_bits < 16:
        with torch.no_grad():
            if w_bits == 1:
                key_scale = (  # supposed to be the absolute mean of activations
                    self_attn.k_norm.weight[None]
                    .expand(
                        self_attn.config.num_key_value_heads,
                        self_attn.config.head_dim,
                    )
                    .clone()
                    .detach()
                )
            elif w_bits == 0 or w_bits == 2:
                key_scale = 2 * (  # supposed to be the absolute max of activations
                    self_attn.k_norm.weight[None]
                    .expand(
                        self_attn.config.num_key_value_heads,
                        self_attn.config.head_dim,
                    )
                    .clone()
                    .detach()
                )
            elif w_bits == 3 or w_bits == 4:
                key_max = 2 * (  # supposed to be the absolute max of activations
                    self_attn.k_norm.weight[None]
                    .expand(
                        self_attn.config.num_key_value_heads,
                        self_attn.config.head_dim,
                    )
                    .clone()
                    .detach()
                )
                maxq = 2 ** (w_bits - 1) - 1
                key_scale = key_max / maxq
            else:
                raise NotImplementedError

            self_attn.key_clip_val = nn.Parameter(
                key_scale.to(self_attn.k_proj.weight.device, self_attn.k_proj.weight.dtype)
            )

    self_attn.forward = QATQwen3Attention.forward.__get__(self_attn, type(self_attn))  # type: ignore
