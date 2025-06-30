from typing import Any, Optional

import torch
from hqq.core.quantize import (  # type: ignore
    BaseQuantizeConfig,  # type: ignore
    HQQLinear,
    Quantizer,
)
from torch import Tensor, nn
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.utils.quantization_config import HqqConfig

from .quantization_util import (
    LsqBinaryTernaryExtension,
    StretchedElasticQuant,
    get_quant_min,
    quantize,
)


class QATLinear(nn.Linear):
    def __init__(
        self,
        *args,
        nbits: int = 16,
        group_size: Optional[int] = None,
        weight_layerwise: bool = False,
    ) -> None:
        super().__init__(*args)
        self.nbits = nbits
        self.group_size = group_size
        self.weight_layerwise = weight_layerwise
        if self.nbits < 16:
            num_blocks = 1 if group_size is None else self.weight.shape[1] // group_size
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0] * num_blocks, 1))

    def forward(self, input: Tensor) -> Tensor:
        assert self.weight.ndim == 2
        real_weights = self.weight
        weight_shape = real_weights.shape
        if self.group_size is not None:
            real_weights = real_weights.view(-1, self.group_size)

        if self.nbits >= 16:
            weight = self.weight
        elif self.nbits == 2 or self.nbits == 0:
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.nbits,
                self.weight_layerwise,
            ).to(input.dtype)  # type: ignore
        elif self.nbits <= 4:
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.nbits,
                self.weight_layerwise,
            ).to(input.dtype)  # type: ignore
        else:
            raise NotImplementedError

        out = nn.functional.linear(input, weight.view(weight_shape))
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


def patch_linear_with_qat_linear(
    linear: nn.Linear,
    nbits: int = 16,
    group_size: Optional[int] = None,
    weight_layerwise: bool = False,
) -> None:
    linear.nbits = nbits  # type: ignore
    linear.group_size = group_size  # type: ignore
    linear.weight_layerwise = weight_layerwise  # type: ignore
    if nbits < 16:
        weight = linear.weight if group_size is None else linear.weight.view(-1, group_size)
        with torch.no_grad():
            if nbits == 1:
                scale = weight.abs().mean(dim=-1, keepdim=True).detach()
            elif nbits == 0 or nbits == 2:
                scale, _ = weight.abs().max(dim=-1, keepdim=True)
            elif nbits == 3 or nbits == 4:
                weight_max, _ = weight.abs().max(dim=-1, keepdim=True)
                max_quant = 2 ** (nbits - 1) - 1
                scale = weight_max / max_quant
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

    quant_min = get_quant_min(nbits)

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
            -torch.tensor(
                quant_min,
                device=hqq_linear.meta["zero"].device,
                dtype=hqq_linear.meta["zero"].dtype,
            )
        )
        new_W_q = Quantizer.pack[hqq_linear.meta["packing"]](  # type: ignore
            (data.view(-1, hqq_group_size) - quant_min).float()
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
    test_equal: bool = True,
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

    if test_equal:
        assert (y1 == y2).all(), f"test_mlp_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."
    else:
        assert torch.allclose(y1, y2), f"test_huggingface_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."


def test_huggingface_quantization(
    nbits: int | float,
    qat_group_size: Optional[int],
    hqq_group_size: Optional[int],
    test_equal: bool = True,
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

    if test_equal:
        assert (y1 == y2).all(), f"test_mlp_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."
    else:
        assert torch.allclose(y1, y2), f"test_huggingface_quantization({nbits=}, {qat_group_size=}, {hqq_group_size=}) failed."
