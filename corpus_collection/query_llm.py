import argparse
import itertools
from typing import Any, Iterator, List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from utils import format_output_df, load_prompts, process_answer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLLM offline query for response generation using datasets"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to the input file containing prompts. Should have a column called 'prompt'. ",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Huggingface model name for the LLM to use for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for response generation.",
    )
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens in each response.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing prompts."
    )

    parser.add_argument(
        "--num_responses_per_prompt",
        type=int,
        default=5,
        help="Number of responses to generate per prompt.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for processing prompts. Useful for slicing the dataset. Default is 0.",
    )
    parser.add_argument(
        "--slice_size",
        type=int,
        default=-1,
        help="Number of prompts to process. If -1, processes all prompts from start_index.",
    )

    return parser.parse_args()


def batched(iterable: Any, n: int) -> Iterator[Tuple[Any, ...]]:
    """Create batches of size n from the iterable."""
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def generate_batched_responses(
    messages: List[str], llm: LLM, sampling_params: SamplingParams, batch_size: int
) -> List[List[Any]]:
    """Generate responses for a list of messages in batches."""
    all_responses = []
    batches = batched(messages, batch_size)
    for batch in tqdm(
        batches, desc="Generating responses", total=len(messages) // batch_size
    ):
        responses = llm.generate(batch, sampling_params=sampling_params, use_tqdm=False)
        all_responses.extend(responses)
    return all_responses


def process_responses(
    responses: List[Any], prompts: List[str], num_responses_per_prompt: int
) -> List[Tuple[str, List[str]]]:
    """Process the responses from the LLM."""
    all_answers = []
    for k, (response, prompt) in enumerate(zip(responses, prompts)):
        answer = []
        for i in range(num_responses_per_prompt):
            if i >= len(response.outputs):
                answer.append("NOT FOUND")
                continue
            answer.append(process_answer(response.outputs[i].text, k, i))
        all_answers.append((prompt, answer))
    return all_answers


def generate_responses(
    prompts: List[str],
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    num_responses_per_prompt: int,
) -> List[List[str]]:
    """Generate responses for a list of prompts."""
    responses = generate_batched_responses(prompts, llm, sampling_params, batch_size)
    processed_responses = process_responses(
        responses, prompts, num_responses_per_prompt
    )
    return processed_responses


def load_llm(
    model: str,
    tensor_parallel_size: int,
    temperature: float,
    max_response_tokens: int,
    num_responses_per_prompt: int,
) -> Tuple[LLM, SamplingParams]:

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_response_tokens,
        n=num_responses_per_prompt,
    )

    if "llama" in model.lower() or "gemma" in model.lower():
        llm = LLM(
            model=model,
            enable_prefix_caching=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes",
            gpu_memory_utilization=0.9,
            max_model_len=(2 * max_response_tokens),
            load_format="bitsandbytes",
        )

    elif "qwen" in model.lower():
        assert "awq" in model.lower(), "Please use the AWQ quantized model for Qwen."
        llm = LLM(
            model=model,
            enable_prefix_caching=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.float16,
            trust_remote_code=True,
            quantization="awq",
            gpu_memory_utilization=0.9,
            max_model_len=(2 * max_response_tokens),
        )
    else:
        raise ValueError("Unsupported model type. Please use a Llama or Qwen model.")

    return llm, sampling_params


def main() -> None:
    """Main function that processes arguments and loads prompts."""
    args = parse_args()
    print(f"Prompts file: {args.prompts}")

    prompts_df, start_index, end_index = load_prompts(
        file_path=args.prompts, start_index=args.start_index, slice_size=args.slice_size
    )

    output_path = args.output + f"_start_index_{start_index}_end_index_{end_index}"
    print(f"Output file: {output_path}")

    prompts = prompts_df["prompt"].tolist()

    llm, sampling_params = load_llm(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_response_tokens=args.max_response_tokens,
        num_responses_per_prompt=args.num_responses_per_prompt,
    )

    processed_responses = generate_responses(
        prompts, llm, sampling_params, args.batch_size, args.num_responses_per_prompt
    )

    prompts_df = format_output_df(
        prompts_df, processed_responses, args.num_responses_per_prompt
    )

    dataset = Dataset.from_pandas(prompts_df)
    print(f"Generated dataset length: {len(dataset)}")

    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
