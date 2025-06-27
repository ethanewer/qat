import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk  # type: ignore
from torch import Tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer  # type: ignore

from paretoq_qat import replace_linear_with_qat_linear


@dataclass
class ModelArguments:
    local_dir: str = field(metadata={"help": "Directory where outputs are saved and loaded from."})
    input_model_filename: str = field(metadata={"help": "Pretrained model identifier or path."})
    output_model_filename: str = field(metadata={"help": "Folder to save the fine-tuned model."})
    train_data_local_path: str = field(metadata={"help": "Path to training data."})
    model_max_length: int = field(default=16384, metadata={"help": "Max sequence length.."})
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT."})
    nbits: int = field(default=4, metadata={"help": "Bit-width for quantized weights."})
    group_size: Optional[int] = field(
        default=None,
        metadata={"help": "Group size for quantized weights."},
    )


def padding_collator(features: list[dict[str, list[int]]]) -> dict[str, Tensor]:
    first = features[0]
    batch = {}
    for k, v in first.items():
        sequences = [torch.tensor(f[k]) for f in features]
        batch[k] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

    return batch


def main():
    parser = HfArgumentParser((ModelArguments, SFTConfig))  # type: ignore
    model_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir = os.path.join(model_args.local_dir, model_args.output_model_filename)
    training_args.remove_unused_columns = False

    model = AutoModelForCausalLM.from_pretrained(model_args.input_model_filename)
    if model_args.qat:
        replace_linear_with_qat_linear(
            model,
            nbits=model_args.nbits,
            group_size=model_args.group_size,
        )

    model.config.use_cache = False

    dataset = load_from_disk(model_args.train_data_local_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=padding_collator,
    )

    trainer.train()
    trainer.save_model(model_args.output_model_filename + "-checkpoint")

    state_dict, _ = get_state_dict(
        trainer.model,
        trainer.optimizer,  # type: ignore
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    trainer.model.module.save_pretrained(  # type: ignore
        os.path.join(model_args.local_dir, model_args.output_model_filename + "-qat"),
        state_dict=state_dict,
        safe_serialization=True,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
