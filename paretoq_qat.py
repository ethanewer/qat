# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
from hqq.core.quantize import HQQLinear, Quantizer  # type: ignore
from torch import Tensor, nn
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
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


class QATLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        w_bits=16,
        weight_layerwise=False,
    ):
        super().__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input: Tensor) -> Tensor:
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

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

        out = nn.functional.linear(input, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


# --------------- NEW ---------------


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
    weight_layerwise: bool = False,
) -> None:
    linear.w_bits = w_bits  # type: ignore
    linear.weight_layerwise = weight_layerwise  # type: ignore
    if w_bits < 16:
        with torch.no_grad():
            if w_bits == 1:
                scale = torch.mean(linear.weight.abs(), dim=-1, keepdim=True).detach()
            elif w_bits == 0 or w_bits == 2:
                scale, _ = torch.max(torch.abs(linear.weight), dim=-1, keepdim=True)
            elif w_bits == 3 or w_bits == 4:
                xmax, _ = torch.max(torch.abs(linear.weight), dim=-1, keepdim=True)
                maxq = 2 ** (w_bits - 1) - 1
                scale = xmax / maxq
            else:
                raise NotImplementedError

            linear.weight_clip_val = nn.Parameter(
                scale.to(linear.weight.device, linear.weight.dtype)
            )

    linear.forward = QATLinear.forward.__get__(linear, type(linear))  # type: ignore


def replace_linear_with_qat_linear(
    module: nn.Module,
    nbits: int = 16,
    weight_layerwise: bool = False,
) -> None:
    if isinstance(module, nn.Linear):
        patch_linear_with_qat_linear(module, nbits, weight_layerwise)
    else:
        for child in list(module.children()):
            replace_linear_with_qat_linear(child, nbits, weight_layerwise)


def update_quantized_linear_inplace(
    hqq_linear: HQQLinear,
    weight: Tensor,
    weight_clip_val: Tensor,
    nbits: int,
    block_size: Optional[int] = None,
) -> None:
    assert isinstance(hqq_linear.meta, dict)
    assert isinstance(hqq_linear.meta["scale"], Tensor)
    assert isinstance(hqq_linear.meta["zero"], Tensor)
    assert isinstance(hqq_linear.W_q, nn.Parameter)

    data, scale = quantize(weight, weight_clip_val, nbits)

    if block_size is None:
        extra_dim = 1
    else:
        extra_dim = data.shape[1] // block_size
        assert extra_dim * block_size == data.shape[1]

    with torch.no_grad():
        new_scale = scale[:, None].expand(scale.shape[0], extra_dim, 1).reshape(-1, 1)
        assert new_scale.shape == hqq_linear.meta["scale"].shape
        hqq_linear.meta["scale"].set_(new_scale)  # type: ignore
        hqq_linear.meta["zero"].fill_(-data.min())
        new_W_q = Quantizer.pack[hqq_linear.meta["packing"]](  # type: ignore
            (data.view(-1, 64) - data.min()).float()
        )
        assert new_W_q.shape == hqq_linear.W_q.shape
        hqq_linear.W_q.set_(new_W_q)


def update_quantized_model_with_qat_state_dict(
    model: nn.Module,
    qat_state_dict: dict[str, Tensor],
    nbits: int,
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, HQQLinear):
            weight = qat_state_dict[name + ".weight"]
            weight_clip_val = qat_state_dict[name + ".weight_clip_val"]
            update_quantized_linear_inplace(module, weight, weight_clip_val, nbits)


def get_quantized_model_from_qat_state_dict(
    qat_state_dict: dict[str, Tensor],
    base_model_name: str,
    nbits: int,
    group_size: Optional[int] = None,
    skip_modules: list[str] = ["lm_head"],
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Any = "cuda",
) -> Any:
    quantized_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=HqqConfig(
            nbits=4,
            group_size=group_size,  # type: ignore
            axis=1,
            skip_modules=skip_modules,
        ),
    )
    update_quantized_model_with_qat_state_dict(quantized_model, qat_state_dict, nbits)
    return quantized_model


def save_qat_model(
    qat_state_dict: dict[str, Tensor],
    base_model_name: str,
    save_path: str,
    nbits: int,
    vllm_compatible: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Any = "cuda",
) -> None:
    quantized_model = get_quantized_model_from_qat_state_dict(
        qat_state_dict=qat_state_dict,
        base_model_name=base_model_name,
        nbits=nbits,
        group_size=64 if vllm_compatible else None,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    quantized_model.save_pretrained(save_path)


# -----------------------------------
