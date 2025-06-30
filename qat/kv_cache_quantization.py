from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from hqq.core.quantize import Quantizer  # type: ignore
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from .quantization_util import (
    LsqBinaryTernaryExtension,
    StretchedElasticQuant,
    get_quant_min,
    quantize,
)


class QATQwen3Attention(Qwen3Attention):
    def __init__(self, *args, nbits: int = 16) -> None:
        super().__init__(*args)
        self.nbits = nbits
        if self.nbits < 16:
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

        cos, sin = position_embeddings
        query_states, real_key_states = apply_rotary_pos_emb(
            query_states,
            real_key_states,
            cos,
            sin,
        )

        if self.nbits >= 16:
            key_states = real_key_states
            value_states = real_value_states
        elif self.nbits == 2 or self.nbits == 0:
            key_states = StretchedElasticQuant.apply(
                real_key_states,
                self.key_clip_val,
                self.nbits,
                False,
            ).to(real_key_states.dtype)  # type: ignore
            value_states = StretchedElasticQuant.apply(
                real_value_states,
                self.value_clip_val,
                self.nbits,
                False,
            ).to(real_value_states.dtype)  # type: ignore
        elif self.nbits <= 4:
            key_states = LsqBinaryTernaryExtension.apply(
                real_key_states,
                self.key_clip_val,
                self.nbits,
                False,
            ).to(real_key_states.dtype)  # type: ignore
            value_states = LsqBinaryTernaryExtension.apply(
                real_value_states,
                self.value_clip_val,
                self.nbits,
                False,
            ).to(real_value_states.dtype)  # type: ignore
        else:
            raise NotImplementedError

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
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


def patch_attention_with_qat_attention(
    self_attn: Qwen3Attention,
    key_states: Tensor,
    value_states: Tensor,
    nbits: int = 16,
) -> None:
    self_attn.nbits = nbits  # type: ignore
    if nbits < 16:
        with torch.no_grad():
            if nbits == 1:
                key_scale = key_states.abs().mean(dim=(0, 1), keepdim=True).detach()
                value_scale = value_states.abs().mean(dim=(0, 1), keepdim=True).detach()
            elif nbits == 0 or nbits == 2:
                key_scale = key_states.abs().max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].detach()
                value_scale = value_states.abs().max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].detach()
            elif nbits == 3 or nbits == 4:
                key_max = key_states.abs().max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].detach()
                value_max = value_states.abs().max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].detach()
                max_quant = 2 ** (nbits - 1) - 1
                key_scale = key_max / max_quant
                value_scale = value_max / max_quant
            else:
                raise NotImplementedError

            self_attn.key_clip_val = nn.Parameter(key_scale.to(self_attn.k_proj.weight.device, self_attn.k_proj.weight.dtype))
            self_attn.value_clip_val = nn.Parameter(
                value_scale.to(self_attn.v_proj.weight.device, self_attn.v_proj.weight.dtype)
            )

    self_attn.forward = QATQwen3Attention.forward.__get__(self_attn, type(self_attn))  # type: ignore


def replace_attention_with_qat_attention(
    model: Qwen3ForCausalLM,
    calibration_inputs: dict[str, Tensor],
    nbits: int = 16,
) -> None:
    past_key_values = DynamicCache()
    with torch.no_grad():
        _ = model(**calibration_inputs, past_key_values=past_key_values, use_cache=True)

    for layer_idx in range(model.config.num_hidden_layers):
        patch_attention_with_qat_attention(
            model.model.layers[layer_idx].self_attn,  # type: ignore
            key_states=past_key_values.key_cache[layer_idx],
            value_states=past_key_values.value_cache[layer_idx],
            nbits=nbits,
        )


def replace_qat_attention_with_attention(model: Qwen3ForCausalLM) -> None:
    for layer in model.model.layers:
        layer.self_attn.forward = Qwen3Attention.forward.__get__(layer.self_attn, type(layer.self_attn))  # type: ignore


class QATQuantizedCache(DynamicCache):
    def __init__(
        self,
        nbits: int,
        qat_model: Qwen3ForCausalLM,
    ) -> None:
        super().__init__()
        self.nbits = nbits
        self.num_key_value_heads = qat_model.config.num_key_value_heads
        self.head_dim = qat_model.config.head_dim
        self.key_meta = [self.get_hqq_meta(layer.self_attn.key_clip_val.detach()) for layer in qat_model.model.layers]  # type: ignore
        self.value_meta = [self.get_hqq_meta(layer.self_attn.value_clip_val.detach()) for layer in qat_model.model.layers]  # type: ignore

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(self.quantize(key_states, self.key_meta[layer_idx]))
                self.value_cache.append(self.quantize(value_states, self.value_meta[layer_idx]))
            elif not self.key_cache[layer_idx].numel():
                self.key_cache[layer_idx] = self.quantize(key_states, self.key_meta[layer_idx])
                self.value_cache[layer_idx] = self.quantize(value_states, self.value_meta[layer_idx])
            else:
                key_states = torch.cat(
                    (self.dequantize(self.key_cache[layer_idx], self.key_meta[layer_idx]), key_states),
                    dim=-2,
                )
                value_states = torch.cat(
                    (self.dequantize(self.value_cache[layer_idx], self.value_meta[layer_idx]), value_states),
                    dim=-2,
                )
                self.key_cache[layer_idx] = self.quantize(key_states, self.key_meta[layer_idx])
                self.value_cache[layer_idx] = self.quantize(value_states, self.value_meta[layer_idx])

        # return key_states, value_states
        print(f"{self.key_cache[layer_idx].shape=}")
        out = (
            self.dequantize(self.key_cache[layer_idx], self.key_meta[layer_idx]),
            self.dequantize(self.value_cache[layer_idx], self.value_meta[layer_idx]),
        )
        print(f"{out[0].shape=}")
        return out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0

        if len(self.key_cache) <= layer_idx:
            return 0

        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def quantize(self, tensor: Tensor, meta: dict) -> Tensor:
        tensor = tensor.transpose(1, 2).view(-1, self.num_key_value_heads * self.head_dim)
        meta["shape_without_padding"] = tensor.shape

        padding = -tensor.shape[0] % 8
        tensor = F.pad(tensor, (0, 0, 0, padding), value=0)

        data, meta["scale"] = quantize(tensor, meta["scale"], self.nbits)
        true_dequantized = (data * meta["scale"]).view(meta["shape"])
        if self.nbits == 2:
            data *= 2
        elif self.nbits == 1:
            data /= 2
        elif self.nbits == 0:
            data /= torch.tensor(1 / 1.5, dtype=data.dtype, device=data.device)

        quant_min = get_quant_min(self.nbits)

        q_tensor = Quantizer.pack[meta["packing"]](
            (data.view(-1, self.num_key_value_heads * self.head_dim) - quant_min).float()
        )

        dequantized = Quantizer.dequantize(q_tensor, meta)
        print(true_dequantized.shape)
        print(dequantized.shape)
        assert torch.allclose(
            dequantized.to(true_dequantized.device, true_dequantized.dtype),
            true_dequantized,
            atol=1e-2,
            rtol=1e-4,
        )

        return q_tensor

    def dequantize(self, q_tensor: Tensor, meta: dict) -> Tensor:
        tensor = Quantizer.dequantize(q_tensor, meta)
        assert tensor.shape[1] == meta["shape_without_padding"][1]
        tensor = tensor[: meta["shape_without_padding"][0]]
        return tensor.view(-1, self._seen_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def get_hqq_meta(self, clip_val: Tensor) -> dict:
        nbits = 1.58 if self.nbits == 0 else self.nbits
        meta = {
            "nbits": nbits,
            "group_size": None,
            "axis": 0,
            "packing": Quantizer.bit_to_packing[nbits],
            "unpack_view_dtype": Quantizer.unpack_view_dtype[Quantizer.bit_to_packing[nbits]],
            "shape": (-1, self.num_key_value_heads * self.head_dim),
            "view_as_float": False,
            "quant_scale": False,
            "quant_zero": False,
            "compute_dtype": torch.bfloat16,
        }

        scale = clip_val.view(1, -1)  # type: ignore
        if nbits == 2:
            scale /= 2
        elif nbits == 1:
            scale *= 2
        elif nbits == 0:
            scale *= torch.tensor(1 / 1.5, dtype=scale.dtype, device=scale.device)

        zero = torch.full_like(scale, -get_quant_min(self.nbits))

        meta["scale"] = scale
        meta["zero"] = zero

        return meta
