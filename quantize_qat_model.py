import argparse
import os

from safetensors.torch import load_file

from paretoq_qat import save_qat_model


def main():
    parser = argparse.ArgumentParser(description="Quantize a model using QAT.")
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base path to the original model directory (e.g., 'local/qwen3-4b/Qwen/Qwen3-4B-4bit')",
    )
    args = parser.parse_args()
    base = args.base

    state_dict = {}
    for fname in sorted(os.listdir(base + "-qat")):
        if fname.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(base + "-qat", fname)))

    save_qat_model(state_dict, "Qwen/Qwen3-4B", base + "-quantized", nbits=4)
