from typing import List, Tuple

import pandas as pd


def read_file(file_path: str) -> pd.DataFrame:
    """Reads a tab-separated file into a pandas DataFrame. Treats the first row as the header."""
    return pd.read_csv(file_path, sep="\t", header=0)


def load_prompts(
    file_path: str, start_index: int = 0, slice_size: int = -1
) -> pd.DataFrame:
    """
    Load prompts from a file for the given start index and slice size.

    Args:
        file_path: Path to the input file containing prompts
        start_index: Start index for processing prompts
        slice_size: Number of prompts to process (-1 for all)

    Returns:
        pd.DataFrame: pandas DataFrame containing the prompts
    """
    prompts_df = read_file(file_path)

    assert "prompt" in prompts_df.columns, "Input file must have a 'prompt' column."

    end_index = start_index + slice_size if slice_size != -1 else len(prompts_df)
    if end_index > len(prompts_df):
        end_index = len(prompts_df)

    print(f"Total prompts in input file: {len(prompts_df)}")
    print(f"Processing prompts from start index (included): {start_index}")
    print(f"Processing prompts up to end index (not included): {end_index}")

    prompts_df = prompts_df.iloc[start_index:end_index]
    print(f"Number of prompts to process: {len(prompts_df)}")

    prompts_df.reset_index(drop=True, inplace=True)

    return prompts_df, start_index, end_index


def format_response_df(
    prompts_df: pd.DataFrame, results: List[Tuple[str, List[str]]], n: int
) -> pd.DataFrame:
    """Add responses to the output dataframe."""
    for i in range(n):
        prompts_df[f"response_{i}"] = None
    for i, row in prompts_df.iterrows():
        prompt_from_df = row["prompt"].strip()
        if i >= len(results):
            continue
        (prompt_from_results, responses) = results[i]
        assert prompt_from_df == prompt_from_results
        assert len(responses) == n
        for j in range(n):
            prompts_df.at[i, f"response_{j}"] = responses[j]
    return prompts_df


def process_answer(answer: str, prompt_number: int, response_number: int) -> str:
    """Process the generated answer from the LLM."""
    if "THE END!" in answer:
        answer = answer.split("THE END!")[0]
    else:
        print(
            f"THE END! not found for prompt {prompt_number} and response {response_number}"
        )
    return answer.strip()
