# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
from torch import Tensor


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
        nbits: int,
        layerwise: bool,
    ) -> Tensor:
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param nbits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.nbits = nbits
        if nbits >= 16:
            return input
        if nbits == 1 or nbits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (nbits - 1))
            Qp = 2 ** (nbits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if nbits == 1:
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
        if ctx.nbits >= 16:
            return grad_outputs, None, None, None

        input, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big  # this is more cpu-friendly than torch.ones(input.shape)
        if ctx.nbits == 1:
            if layerwise:
                grad_alpha = ((input.sign()) * grad_outputs * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = (input.sign()) * grad_outputs * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round()))
                        * grad_outputs
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round()))
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
        nbits: int,
        layerwise: bool,
    ) -> Tensor:
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param nbits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.nbits = nbits
        if nbits >= 16:
            return input
        if nbits == 1 or nbits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (nbits - 1))
            Qp = 2 ** (nbits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if nbits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (nbits - 1)
            shift = 0.5  # type: ignore
        Qp = (n_levels - shift) / n_levels  # type: ignore
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if nbits == 1:
            q_w = input.sign()
        else:
            q_w = (torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(  # type: ignore
        ctx: Any,
        grad_outputs: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], None, None]:
        if ctx.nbits >= 16:
            return grad_outputs, None, None, None

        input, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input / alpha
        clip_val = 1 - 1e-2
        if ctx.nbits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.nbits - 1)
            shift = 0.5  # type: ignore
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.nbits == 1:
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
                                + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
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
                        * (-q_w + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels)
                    )
                    * grad_outputs
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_outputs
        return grad_input, grad_alpha, None, None


def quantize_lsq_binary_ternary_extension(
    input: Tensor,
    alpha: Tensor,
    nbits: int,
) -> tuple[Tensor, Tensor]:
    if nbits >= 16:
        raise NotImplementedError

    if nbits == 1 or nbits == 0:
        Qn = -1
        Qp = 1
    else:
        Qn = -(2 ** (nbits - 1))
        Qp = 2 ** (nbits - 1) - 1

    eps = torch.tensor(0.00001, device=alpha.device).float()

    alpha = torch.where(alpha > eps, alpha, eps)

    if nbits == 1:
        q_w = input.sign()
    else:
        q_w = (input / alpha).round().clamp(Qn, Qp)

    return q_w, alpha


def quantize_stretched_elastic_quant(
    input: Tensor,
    alpha: Tensor,
    nbits: int,
) -> tuple[Tensor, Tensor]:
    if nbits >= 16:
        raise NotImplementedError

    eps = torch.tensor(0.00001, device=alpha.device).float()
    alpha = torch.where(alpha > eps, alpha, eps)

    clip_val = 1 - 1e-2
    if nbits == 0:
        n_levels = 1.5
        shift = 0.0
    else:
        n_levels = 2 ** (nbits - 1)
        shift = 0.5

    if nbits == 1:
        q_w = input.sign()
    else:
        q_w = (torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels

    return q_w, alpha


def quantize(weight: Tensor, weight_clip_val: Tensor, nbits: int) -> tuple[Tensor, Tensor]:
    if nbits == 2 or nbits == 0:
        return quantize_stretched_elastic_quant(
            weight,
            weight_clip_val,
            nbits,
        )
    elif nbits <= 4:
        return quantize_lsq_binary_ternary_extension(
            weight,
            weight_clip_val,
            nbits,
        )
    else:
        raise NotImplementedError


def get_quant_min(nbits: int) -> float:
    if nbits == 0 or nbits == 2:
        clip_val = 1 - 1e-2
        if nbits == 0:
            n_levels = 1.5
            shift = 0.0
        else:
            n_levels = 2 ** (nbits - 1)
            shift = 0.5

        return round(-clip_val * n_levels - shift) + shift
    elif nbits == 1:
        return -0.5
    elif nbits <= 4:
        return -(2 ** (nbits - 1)) + 1
    else:
        raise NotImplementedError
