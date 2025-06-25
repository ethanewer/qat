import os
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk  # type: ignore
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from trl import SFTConfig, SFTTrainer  # type: ignore

from paretoq_qat import replace_linear_with_qat_linear, save_qat_model


@dataclass
class ModelArguments:
    local_dir: str = field(metadata={"help": "Directory where outputs are saved and loaded from."})
    input_model_filename: str = field(metadata={"help": "Pretrained model identifier or path."})
    output_model_filename: str = field(metadata={"help": "Folder to save the fine-tuned model."})
    train_data_local_path: str = field(metadata={"help": "Path to training data."})
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT."})
    nbits: int = field(default=4, metadata={"help": "Bit-width for QAT weight quantization."})
    model_max_length: int = field(default=16384, metadata={"help": "Max sequence length.."})


def main():
    parser = HfArgumentParser((ModelArguments, SFTConfig))  # type: ignore
    model_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir = os.path.join(model_args.local_dir, model_args.output_model_filename)

    model = AutoModelForCausalLM.from_pretrained(model_args.input_model_filename)
    if model_args.qat:
        replace_linear_with_qat_linear(model, nbits=model_args.nbits)

    model.config.use_cache = False

    dataset = load_from_disk(model_args.train_data_local_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        model_max_length=model_args.model_max_length,
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_args.output_model_filename)

    state_dict, _ = get_state_dict(
        trainer.model,
        trainer.optimizer,  # type: ignore
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    save_qat_model(
        qat_state_dict=state_dict,  # type: ignore
        base_model_name=model_args.input_model_filename,
        save_path=os.path.join(model_args.local_dir, model_args.output_model_filename),
        nbits=model_args.nbits,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
