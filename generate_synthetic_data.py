import argparse

from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    load_dataset,  # type: ignore
)
from sklearn.model_selection import train_test_split  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer
from vllm import LLM, SamplingParams  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qwen3 distillation dataset")
    parser.add_argument(
        "--model-size",
        type=str,
        required=True,
        help="Size if Qwen3 model.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process from each split (None = all)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=512,
        help="Number of test examples.",
    )
    return parser.parse_args()


def make_dataset(data: list[dict]) -> Dataset:
    input_ids = []
    attention_mask = []
    labels = []
    for example in data:
        prompt_ids = example["input_ids"]
        response_ids = example["output_ids"]
        input_ids.append(prompt_ids + response_ids)
        attention_mask.append([1] * len(input_ids[-1]))
        labels.append([-100] * len(prompt_ids) + response_ids)

    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    )


def main():
    args = parse_args()
    print(args)

    model_name = f"Qwen/Qwen3-{args.model_size}B"
    print(f"{model_name=}")

    ds = load_dataset("nvidia/AceReason-Math", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted_inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": problem}] for problem in ds["problem"]],  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    if args.num_examples is not None:
        formatted_inputs = formatted_inputs[: args.num_examples]

    model = LLM(
        model=model_name,
        tensor_parallel_size=4,
        max_num_seqs=256,
        max_num_batched_tokens=131072,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(max_tokens=16384)

    responses = model.generate(
        prompts=formatted_inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    data = [
        {
            "input_text": response.prompt,
            "input_ids": response.prompt_token_ids,
            "output_text": response.outputs[0].text,
            "output_ids": response.outputs[0].token_ids,
        }
        for response in responses
    ]

    train_data, eval_data = train_test_split(
        data,
        test_size=args.test_size if args.test_size < 1 else int(args.test_size),
        random_state=0,
    )

    dataset = DatasetDict(
        {
            "train": make_dataset(train_data),
            "eval": make_dataset(eval_data),
        }
    )

    dataset.save_to_disk(f"local/qwen3-{args.model_size}b-dataset")


if __name__ == "__main__":
    main()
