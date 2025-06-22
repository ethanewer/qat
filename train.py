import argparse
import os
from argparse import Namespace

import torch
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor
from torch import distributed as dist
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from paretoq_qat import replace_linear_with_quantized_linear


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ex = self.examples[idx]
        prompt_ids = ex["input_ids"]
        response_ids = ex["output_ids"]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def parse_args() -> Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", type=str, required=True)
    p.add_argument("--input_model_filename", type=str, required=True)
    p.add_argument("--output_model_filename", type=str, required=True)
    p.add_argument("--train_data_local_path", type=str, required=True)
    p.add_argument("--num_eval", type=int, required=False, default=100)
    p.add_argument("--do_train", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--do_eval", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--tf32", type=lambda x: x.lower() == "true", default=True)
    p.add_argument(
        "--gradient_checkpointing",
        type=lambda x: x.lower() == "true",
        default=False,
    )
    p.add_argument("--qat", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--w_bits", type=int, default=4)
    p.add_argument(
        "--log_on_each_node",
        type=lambda x: x.lower() == "true",
        default=False,
    )
    p.add_argument("--logging_dir", type=str, default=None)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--evaluation_strategy", type=str, default="no")
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=1)
    return p.parse_args()


def get_training_arguments(args: Namespace) -> TrainingArguments:
    return TrainingArguments(  # type: ignore
        output_dir=os.path.join(args.local_dir, args.output_model_filename),
        do_train=args.do_train,
        do_eval=args.do_eval,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        disable_tqdm=not args.log_on_each_node,
        remove_unused_columns=False,
    )


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")

    model = AutoModelForCausalLM.from_pretrained(args.input_model_filename)
    replace_linear_with_quantized_linear(model, w_bits=args.w_bits)
    model.config.use_cache = False

    data = torch.load(args.train_data_local_path)
    train_data, eval_data = train_test_split(
        data, test_size=args.num_eval, random_state=0
    )
    train_dataset = SFTDataset(train_data)
    eval_dataset = SFTDataset(eval_data)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=AutoTokenizer.from_pretrained(args.model_name),
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=get_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
