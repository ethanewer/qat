import argparse
import os

from safetensors.torch import load_file

from paretoq_qat import get_quantized_model_from_qat_state_dict


def main():
    parser = argparse.ArgumentParser(description="Quantize a model using QAT.")
    parser.add_argument(
        "--model-size",
        type=str,
        default="4",
        help="Size if Qwen3 model.",
    )
    parser.add_argument(
        "--qat-nbits",
        type=int,
        required=True,
        help="Bit-width for quantized weights.",
    )
    parser.add_argument(
        "--hqq-nbits",
        type=int,
        required=True,
        help="Bit-width for quantized weights.",
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
    base = f"local/qwen3-{args.model_size}b/Qwen/Qwen3-{args.model_size}B-{args.nbits}bit"

    state_dict = {}
    for fname in sorted(os.listdir(base + "-qat")):
        if fname.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(base + "-qat", fname)))

    quantized_model = get_quantized_model_from_qat_state_dict(
        state_dict,
        f"Qwen/Qwen3-{args.model_size}B",
        qat_nbits=args.qat_nbits,
        hqq_nbits=args.hqq_nbits,
        qat_group_size=args.qat_group_size,
        hqq_group_size=args.hqq_group_size,
    )

    quantized_model.save_pretrained(base + "-quantized")


if __name__ == "__main__":
    main()
