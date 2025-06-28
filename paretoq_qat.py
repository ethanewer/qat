# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, Optional

import torch
from hqq.core.quantize import (  # type: ignore
    BaseQuantizeConfig,  # type: ignore
    HQQLinear,
    Quantizer,
)
from torch import Tensor, nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)
from transformers.utils.quantization_config import HqqConfig


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        alpha: Tensor,
        num_bits: int,
        layerwise: bool,
    ) -> Tensor:
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(  # type: ignore
        ctx: Any,
        grad_outputs: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], None, None]:
        if ctx.num_bits >= 16:
            return grad_outputs, None, None, None

        input, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = ((input.sign()) * grad_outputs * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = (input.sign()) * grad_outputs * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_outputs
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_outputs
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_outputs
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        alpha: Tensor,
        num_bits: int,
        layerwise: bool,
    ) -> Tensor:
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5  # type: ignore
        Qp = (n_levels - shift) / n_levels  # type: ignore
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift)
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(  # type: ignore
        ctx: Any,
        grad_outputs: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], None, None]:
        if ctx.num_bits >= 16:
            return grad_outputs, None, None, None

        input, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5  # type: ignore
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = ((input.sign()) * grad_outputs * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = (input.sign()) * grad_outputs * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_outputs
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_outputs
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_outputs
        return grad_input, grad_alpha, None, None


# --------------- NEW ---------------


class QATLinear(nn.Linear):
    def __init__(
        self,
        *args,
        w_bits: int = 16,
        group_size: Optional[int] = None,
        weight_layerwise: bool = False,
    ) -> None:
        super().__init__(*args)
        self.w_bits = w_bits
        self.group_size = group_size
        self.weight_layerwise = weight_layerwise
        if self.w_bits < 16:
            num_blocks = 1 if group_size is None else self.weight.shape[1] // group_size
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0] * num_blocks, 1))

    def forward(self, input: Tensor) -> Tensor:
        assert self.weight.ndim == 2
        real_weights = self.weight
        weight_shape = real_weights.shape
        if self.group_size is not None:
            real_weights = real_weights.view(-1, self.group_size)

        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits == 2 or self.w_bits == 0:
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input.dtype)  # type: ignore
        elif self.w_bits <= 4:
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input.dtype)  # type: ignore
        else:
            raise NotImplementedError

        out = nn.functional.linear(input, weight.view(weight_shape))
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


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


def quantize_lsq_binary_ternary_extension(
    input: Tensor,
    alpha: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    if num_bits >= 16:
        raise NotImplementedError

    if num_bits == 1 or num_bits == 0:
        Qn = -1
        Qp = 1
    else:
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1

    eps = torch.tensor(0.00001, device=alpha.device).float()

    alpha = torch.where(alpha > eps, alpha, eps)

    if num_bits == 1:
        q_w = input.sign()
    else:
        q_w = (input / alpha).round().clamp(Qn, Qp)

    return q_w, alpha


def quantize_stretched_elastic_quant(
    input: Tensor,
    alpha: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    if num_bits >= 16:
        raise NotImplementedError

    eps = torch.tensor(0.00001, device=alpha.device).float()
    alpha = torch.where(alpha > eps, alpha, eps)

    clip_val = 1 - 1e-2
    if num_bits == 0:
        n_levels = 1.5
        shift = 0
    else:
        n_levels = 2 ** (num_bits - 1)
        shift = 0.5  # type: ignore

    if num_bits == 1:
        q_w = input.sign()
    else:
        q_w = (
            torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift
        ) / n_levels

    return q_w, alpha


def quantize(weight: Tensor, weight_clip_val: Tensor, w_bits: int) -> tuple[Tensor, Tensor]:
    if w_bits == 2 or w_bits == 0:
        return quantize_stretched_elastic_quant(
            weight,
            weight_clip_val,
            w_bits,
        )
    elif w_bits <= 4:
        return quantize_lsq_binary_ternary_extension(
            weight,
            weight_clip_val,
            w_bits,
        )
    else:
        raise NotImplementedError


def patch_linear_with_qat_linear(
    linear: nn.Linear,
    w_bits: int = 16,
    group_size: Optional[int] = None,
    weight_layerwise: bool = False,
) -> None:
    linear.w_bits = w_bits  # type: ignore
    linear.group_size = group_size  # type: ignore
    linear.weight_layerwise = weight_layerwise  # type: ignore
    if w_bits < 16:
        weight = linear.weight if group_size is None else linear.weight.view(-1, group_size)
        with torch.no_grad():
            if w_bits == 1:
                scale = torch.mean(weight.abs(), dim=-1, keepdim=True).detach()
            elif w_bits == 0 or w_bits == 2:
                scale, _ = torch.max(torch.abs(weight), dim=-1, keepdim=True)
            elif w_bits == 3 or w_bits == 4:
                xmax, _ = torch.max(torch.abs(weight), dim=-1, keepdim=True)
                maxq = 2 ** (w_bits - 1) - 1
                scale = xmax / maxq
            else:
                raise NotImplementedError

            linear.weight_clip_val = nn.Parameter(scale.to(weight.device, weight.dtype))

    linear.forward = QATLinear.forward.__get__(linear, type(linear))  # type: ignore


def replace_linear_with_qat_linear(
    module: nn.Module,
    nbits: int = 16,
    group_size: Optional[int] = None,
    weight_layerwise: bool = False,
) -> None:
    if isinstance(module, nn.Linear):
        patch_linear_with_qat_linear(module, nbits, group_size, weight_layerwise)
    else:
        for child in list(module.children()):
            replace_linear_with_qat_linear(child, nbits, group_size, weight_layerwise)


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

    self_attn.forward = QATLinear.forward.__get__(self_attn, type(self_attn))  # type: ignore


def update_quantized_linear_inplace(
    hqq_linear: HQQLinear,
    weight: Tensor,
    weight_clip_val: Tensor,
    nbits: int,
    qat_group_size: Optional[int] = None,
    hqq_group_size: Optional[int] = None,
) -> None:
    assert isinstance(hqq_linear.meta, dict)
    assert isinstance(hqq_linear.meta["scale"], Tensor)
    assert isinstance(hqq_linear.meta["zero"], Tensor)
    assert isinstance(hqq_linear.W_q, nn.Parameter)

    if qat_group_size is None:
        qat_group_size = weight.shape[1]

    if hqq_group_size is None:
        hqq_group_size = weight.shape[1]

    assert qat_group_size % hqq_group_size == 0

    data, scale = quantize(weight.view(-1, qat_group_size), weight_clip_val, nbits)
    true_dequantized = (data * scale).view(hqq_linear.meta["shape"])  # type: ignore
    if nbits == 2:
        data *= 2
        scale /= 2
    elif nbits == 1:
        data /= 2
        scale *= 2
    elif nbits == 0:
        data /= torch.tensor(1 / 1.5, dtype=data.dtype, device=data.device)
        scale *= torch.tensor(1 / 1.5, dtype=scale.dtype, device=scale.device)

    with torch.no_grad():
        extra_dim = qat_group_size // hqq_group_size
        new_scale = scale[:, None].expand(scale.shape[0], extra_dim, 1).reshape(-1, 1)
        assert new_scale.shape == hqq_linear.meta["scale"].shape
        hqq_linear.meta["scale"].set_(
            new_scale.to(  # type: ignore
                hqq_linear.meta["scale"].device,
                hqq_linear.meta["scale"].dtype,
            )
        )
        hqq_linear.meta["zero"].fill_(
            -data.min().to(
                hqq_linear.meta["zero"].device,
                hqq_linear.meta["zero"].dtype,
            )
        )
        new_W_q = Quantizer.pack[hqq_linear.meta["packing"]](  # type: ignore
            (data.view(-1, hqq_group_size) - data.min()).float()
        )

        assert new_W_q.shape == hqq_linear.W_q.shape
        hqq_linear.W_q.set_(
            new_W_q.to(
                hqq_linear.W_q.device,
                hqq_linear.W_q.dtype,
            )
        )
        dequantized = Quantizer.dequantize(hqq_linear.W_q, hqq_linear.meta)
        assert torch.allclose(
            dequantized.to(true_dequantized.device, true_dequantized.dtype),
            true_dequantized,
            atol=1e-2,
            rtol=1e-4,
        )


def update_quantized_model_with_qat_state_dict(
    model: nn.Module,
    qat_state_dict: dict[str, Tensor],
    nbits: int,
    qat_group_size: Optional[int] = None,
    hqq_group_size: Optional[int] = None,
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, HQQLinear):
            update_quantized_linear_inplace(
                hqq_linear=module,
                weight=qat_state_dict[name + ".weight"],
                weight_clip_val=qat_state_dict[name + ".weight_clip_val"],
                nbits=nbits,
                qat_group_size=qat_group_size,
                hqq_group_size=hqq_group_size,
            )


def replace_linear_with_hqq_linear(
    model: nn.Module,
    quant_config: dict,
    device: str,
    compute_dtype: torch.dtype,
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            setattr(
                parent,
                name.split(".")[-1],
                HQQLinear(module, quant_config, device=device, compute_dtype=compute_dtype),
            )


def get_quantized_model_from_qat_state_dict(
    qat_state_dict: dict[str, Tensor],
    base_model_name: str,
    nbits: int,
    qat_group_size: Optional[int] = None,
    hqq_group_size: Optional[int] = None,
    skip_modules: list[str] = ["lm_head"],
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Any = "cuda",
) -> Any:
    quantized_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=HqqConfig(
            nbits=1.58 if nbits == 0 else nbits,  # type: ignore
            group_size=hqq_group_size,  # type: ignore
            axis=1,
            skip_modules=skip_modules,
        ),
    )
    update_quantized_model_with_qat_state_dict(
        model=quantized_model,
        qat_state_dict=qat_state_dict,
        nbits=nbits,
        qat_group_size=qat_group_size,
        hqq_group_size=hqq_group_size,
    )
    return quantized_model


def test_mlp_quantization(
    nbits: int | float,
    qat_group_size: Optional[int],
    hqq_group_size: Optional[int],
) -> None:
    model1 = torch.nn.Sequential(torch.nn.Linear(512, 256, bias=False)).to(torch.bfloat16)
    replace_linear_with_qat_linear(
        model1,
        nbits=0 if nbits == 1.58 else nbits,  # type: ignore
        group_size=qat_group_size,
    )

    model2 = torch.nn.Sequential(torch.nn.Linear(512, 256, bias=False)).to(torch.bfloat16)
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=hqq_group_size)  # type: ignore
    quant_config["weight_quant_params"]["optimize"] = False
    replace_linear_with_hqq_linear(model2, quant_config, device="cpu", compute_dtype=torch.bfloat16)
    update_quantized_model_with_qat_state_dict(
        model2,
        model1.state_dict(),
        nbits=0 if nbits == 1.58 else nbits,  # type: ignore
        qat_group_size=qat_group_size,
        hqq_group_size=hqq_group_size,
    )

    x = torch.randn(1, 512, dtype=torch.bfloat16)

    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)

    assert torch.allclose(y1, y2), (
        f"test_mlp_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."
    )


def test_huggingface_quantization(
    nbits: int | float,
    qat_group_size: Optional[int],
    hqq_group_size: Optional[int],
) -> None:
    model1 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    replace_linear_with_qat_linear(
        model1.model,
        nbits=0 if nbits == 1.58 else nbits,  # type: ignore
        group_size=qat_group_size,
    )

    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=hqq_group_size)  # type: ignore
    quant_config["weight_quant_params"]["optimize"] = False
    replace_linear_with_hqq_linear(
        model2.model,
        quant_config,
        device="cpu",
        compute_dtype=torch.bfloat16,
    )
    update_quantized_model_with_qat_state_dict(
        model2,
        model1.state_dict(),
        nbits=0 if nbits == 1.58 else nbits,  # type: ignore
        qat_group_size=qat_group_size,
        hqq_group_size=hqq_group_size,
    )

    input_ids = torch.arange(32)[None]

    with torch.no_grad():
        y1 = model1(input_ids).logits
        y2 = model2(input_ids).logits

    assert torch.allclose(y1, y2), (
        f"test_huggingface_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."
    )


# -----------------------------------
