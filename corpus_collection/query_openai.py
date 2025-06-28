import argparse
import asyncio
import logging
import os
from typing import List, Tuple

import aiolimiter
import openai
import pandas as pd
from datasets import Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from utils import format_output_df, load_prompts, process_answer

# Reference:
# https://github.com/zeno-ml/zeno-build/blob/3a5dcb2ed8bfdeec0dd2bacf2bea76673399824d/zeno_build/models/providers/openai_utils.py#L118

# Edit: Recent changes (11-15-2023) prompted by various changes to the OpenAI API.

client = AsyncOpenAI()

ERROR_ERRORS_TO_MESSAGES = {
    openai.AuthenticationError: "Authentication Error: {e}. Usually due to incorrect OPENAI_API_KEY.",
    openai.BadRequestError: "Bad Request Error: {e}. Usually due to missing or invalid parameters.",
    openai.PermissionDeniedError: "Permission Denied Error: {e}. You don't have access to the requested resource.",
    openai.NotFoundError: "Not Found Error: {e}. The requested resource doesn't exist.",
    openai.APIConnectionError: "API Connection Error: {e}. There was a problem connecting to the OpenAI API.",
    openai.APITimeoutError: "API Timeout Error: {e}. The request timed out.",
    openai.InternalServerError: "Internal Server Error: {e}. An error occurred on the server side.",
    openai.RateLimitError: "Rate Limit Error: {e}. You've hit the OpenAI API rate limit.",
    openai.UnprocessableEntityError: "Unprocessable Entity Error: {e}. Unable to process the request despite the format being correct.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI query for response generation using datasets"
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
        default="gpt-4o-mini-2024-07-18",
        help="OpenAI model name for the LLM to use for generation.",
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
        "--num_requests_per_minute",
        type=int,
        default=150,
        help="Number of requests per minute.",
    )

    parser.add_argument(
        "--num_responses_per_prompt",
        type=int,
        default=5,
        help="Number of responses to generate per prompt.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for the prompts to process.",
    )
    parser.add_argument(
        "--slice_size",
        type=int,
        default=-1,
        help="Slice size for the prompts to process.",
    )

    return parser.parse_args()


def format_prompt(prompt) -> List[dict]:
    return [{"role": "user", "content": prompt}]


async def _throttled_openai_chat_completion_acreate(
    model,
    messages,
    temperature,
    max_tokens,
    limiter: aiolimiter.AsyncLimiter,
    num_responses_per_prompt,
):
    async with limiter:
        for _ in range(3):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    n=num_responses_per_prompt,
                )
            except (
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
            except openai.UnprocessableEntityError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(10)
            except openai.RateLimitError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(60)
            except Exception as e:
                logging.warning(e)
            await asyncio.sleep(30)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages_list,
    model,
    temperature,
    max_tokens,
    requests_per_minute,
    num_responses_per_prompt,
) -> List[List[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        prompts: Prompts.
        model: Model.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        requests_per_minute: Number of requests per minute to allow.
        num_responses_per_prompt: Number of responses to generate per prompt.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            limiter=limiter,
            num_responses_per_prompt=num_responses_per_prompt,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return responses


def generate_responses(
    prompts,
    model,
    temperature,
    max_tokens,
    requests_per_minute,
    num_responses_per_prompt,
) -> List[Tuple[str, List[str]]]:
    messages_list = [format_prompt(prompt) for prompt in prompts]
    predictions = asyncio.run(
        generate_from_openai_chat_completion(
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            requests_per_minute=requests_per_minute,
            num_responses_per_prompt=num_responses_per_prompt,
        )
    )
    assert len(predictions) == len(prompts)
    results = []
    for k, (prompt, prediction) in enumerate(zip(prompts, predictions)):
        result = []
        for x in range(num_responses_per_prompt):
            if x >= len(prediction.choices):
                result.append("")
                continue
            result.append(process_answer(prediction.choices[x].message.content, k, x))
        results.append((prompt, result))
    return results


def main() -> None:
    args = parse_args()
    print(f"Prompts file: {args.prompts}")
    print(f"Start index: {args.start_index}")
    print(f"Slice size: {args.slice_size}")

    prompts_df, start_index, end_index = load_prompts(
        file_path=args.prompts, start_index=args.start_index, slice_size=args.slice_size
    )

    output_path = args.output + f"_start_index_{start_index}_end_index_{end_index}"
    print(f"Output file: {output_path}")

    prompts = prompts_df["prompt"].tolist()
    print(f"Number of prompts: {len(prompts)}")

    results = generate_responses(
        prompts,
        args.model,
        args.temperature,
        args.max_response_tokens,
        args.num_requests_per_minute,
        args.num_responses_per_prompt,
    )

    results_df = format_output_df(prompts_df, results, args.num_responses_per_prompt)
    print(f"Results dataframe: {results_df.head()}")
    dataset = Dataset.from_pandas(results_df)
    print(f"Generated dataset length: {len(dataset)}")

    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
