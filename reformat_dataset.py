import torch
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


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


data = torch.load("local/qwen3_4b_data.pt")
train_data, eval_data = train_test_split(
    data,
    test_size=256,
    random_state=0,
)

dataset = DatasetDict(
    {
        "train": make_dataset(train_data),
        "eval": make_dataset(eval_data),
    }
)

dataset.save_to_disk("local/qwen3_4b_dataset")
