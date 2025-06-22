import argparse
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qwen3-4B distillation dataset")
    parser.add_argument(
        "--num-examples", type=int, default=None,
        help="Maximum number of examples to process from each split (None = all)"
    )
    parser.add_argument(
        "--output-file", type=str, default="qwen3_4b_data.pt",
        help="Path to save the output PyTorch file"
    )
    return parser.parse_args()


def process_response(response):
    return {
        "input_text": response.prompt,
        "input_ids": response.prompt_token_ids,
        "output_text": responses[0].outputs[0].text,
        "output_ids": rresponses[0].outputs[0].token_ids,
    }   


def main():
    args = parse_args()

    ds = load_dataset("nvidia/AceReason-Math", split="train")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    formatted_inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": problem}] for problem in ds["problem"]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    if args.num_examples is not None:
        formatted_inputs = formatted_inputs[:args.num_example]

    model = LLM(
        model="Qwen/Qwen3-4B", 
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

    processed_responses = [
        {
            "input_text": response.prompt,
            "input_ids": response.prompt_token_ids,
            "output_text": responses[0].outputs[0].text,
            "output_ids": responses[0].outputs[0].token_ids,
        }   
        for response in responses
    ]
    
    torch.save(processed_responses, args.output_file)
    print(f"Saved {len(processed_responses)} examples to {args.output_file}")

if __name__ == "__main__":
    main()
