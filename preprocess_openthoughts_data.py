import argparse
from functools import partial

from datasets import DatasetDict, load_dataset  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen3 distillation dataset from OpenThoughts3."
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=None,
        help="Maximum number of train to process from each split (None = all)",
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=512,
        help="Number of eval examples.",
    )
    return parser.parse_args()


def preprocess_batch(
    batch: dict[str, list[str]],
    tokenizer: Qwen2TokenizerFast,
) -> dict[str, list[list[int]]]:
    batch_prompt_ids: list[list[int]] = tokenizer.apply_chat_template(  # type: ignore
        [[{"role": "user", "content": instruction}] for instruction in batch["instruction_seed"]],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    batch_output_ids: list[list[int]] = tokenizer(batch["output"])["input_ids"]  # type: ignore

    input_ids = []
    labels = []
    attention_mask = []
    for prompt_ids, output_ids in zip(batch_prompt_ids, batch_output_ids):
        input_ids.append(prompt_ids + output_ids)
        labels.append([-100] * len(prompt_ids) + output_ids)
        attention_mask.append([1] * len(input_ids[-1]))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    ds = load_dataset("mlfoundations-dev/OpenThoughts3", split="train").filter(
        lambda outputs: [output is not None for output in outputs],
        batched=True,
        batch_size=1000,
        input_columns=["output"],
    )

    num_eval_examples = args.num_eval_examples
    if args.num_train_examples is None:
        num_train_examples = len(ds) - args.num_eval_examples  # type: ignore
    else:
        num_train_examples = args.num_train_examples

    train_ds = ds.select(range(num_train_examples)).map(  # type: ignore
        partial(preprocess_batch, tokenizer=tokenizer),
        remove_columns=ds.column_names,  # type: ignore
        batched=True,
        batch_size=1000,
    )

    if num_eval_examples is None:
        eval_ds = None
    else:
        eval_ds = ds.select(range(num_train_examples, num_train_examples + num_eval_examples)).map(  # type: ignore
            partial(preprocess_batch, tokenizer=tokenizer),
            remove_columns=ds.column_names,  # type: ignore
            batched=True,
            batch_size=1000,
        )

    preprocessed_ds = DatasetDict({"train": train_ds, "eval": eval_ds})
    preprocessed_ds.save_to_disk("local/qwen3-openthoughts-dataset")


if __name__ == "__main__":
    main()
