import argparse
import os

from safetensors.torch import load_file

from paretoq_qat import save_qat_model


def main():
    parser = argparse.ArgumentParser(description="Quantize a model using QAT.")
    parser.add_argument(
        "--model-size",
        type=str,
        default="4",
        help="Size if Qwen3 model.",
    )
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base path to the original model directory (e.g., 'local/qwen3-4b/Qwen/Qwen3-4B-4bit')",
    )
    parser.add_argument(
        "--qat-group-size",
        type=int,
        default=128,
        help="Group size of QAT model's quantized weights.",
    )
    parser.add_argument(
        "--hqq-group-size",
        type=int,
        default=64,
        help="Group size of HQQ model's quantized weights (must be 64 for vLLM compatibility).",
    )

    args = parser.parse_args()
    base = f"local/qwen3-4b/Qwen/Qwen3-{args.model_size}B-{args.nbits}bit"

    state_dict = {}
    for fname in sorted(os.listdir(base + "-qat")):
        if fname.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(base + "-qat", fname)))

    save_qat_model(
        state_dict,
        f"Qwen/Qwen3-{args.model_size}B",
        base + "-quantized",
        nbits=args.nbits,
        qat_group_size=args.qat_group_size,
        hqq_group_size=args.hqq_group_size,
    )
