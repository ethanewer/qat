import os
from dataclasses import dataclass, field

import torch
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.hf_argparser import HfArgumentParser
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


@dataclass
class ModelArguments:
    """Custom arguments for model & data paths and QAT settings."""

    local_dir: str = field(
        metadata={"help": "Directory where outputs are saved and loaded from."}
    )
    input_model_filename: str = field(
        metadata={"help": "Pretrained model identifier or path."}
    )
    output_model_filename: str = field(
        metadata={"help": "Folder name under `local_dir` to save the fine-tuned model."}
    )
    train_data_local_path: str = field(
        metadata={"help": "Path to the serialized training data (torch.save)."}
    )
    num_eval: int = field(
        default=100, metadata={"help": "Number of examples to hold out for eval."}
    )
    qat: bool = field(
        default=False,
        metadata={"help": "Whether to apply quantization-aware training."},
    )
    w_bits: int = field(
        default=4, metadata={"help": "Bit-width for QAT weight quantization."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))  # type: ignore
    model_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir = os.path.join(
        model_args.local_dir, model_args.output_model_filename
    )

    torch.distributed.init_process_group(backend="nccl")

    model = AutoModelForCausalLM.from_pretrained(model_args.input_model_filename)
    if model_args.qat:
        replace_linear_with_quantized_linear(model, w_bits=model_args.w_bits)

    model.config.use_cache = False

    data = torch.load(model_args.train_data_local_path)
    train_data, eval_data = train_test_split(
        data,
        test_size=model_args.num_eval,
        random_state=0,
    )
    train_dataset = SFTDataset(train_data)
    eval_dataset = SFTDataset(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(model_args.input_model_filename)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
