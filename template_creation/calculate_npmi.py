import argparse
import json
import math
import string
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from utils import load_responses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate NPMI between words and nationalities"
    )
    parser.add_argument(
        "--input_file",
        help="Path to input dataset file. use tokenized responses",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_file", help="Path to output JSON file", required=True, type=str
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of tokenized columns to process (0 to n-1)",
        default=5,
    )

    args = parser.parse_args()

    return args


def calculate_npmi(dataset, n_columns):
    """
    Calculate NPMI between words and nationalities.

    Args:
        dataset: HuggingFace dataset with tokenized response columns and nationality column
        n_columns: Number of tokenized columns to process (0 to n-1)

    Returns:
        dict: {nationality: {word: npmi_score}} sorted by descending NPMI scores
    """
    # Count co-occurrences
    word_nationality_count = defaultdict(int)  # (word, nationality) -> count
    word_count = defaultdict(
        int
    )  # word -> total count for that word across all nationalities
    nationality_count = defaultdict(
        int
    )  # nationality -> total word count from that nationality across all words
    total_word_count = 0

    print("Processing dataset to count word-nationality co-occurrences...")

    # Progress bar for processing examples
    for example in tqdm(dataset, desc="Processing examples", unit="examples"):
        nationality = example["nationality"]

        # Collect all words from all tokenized response columns (0 to n-1)
        all_words = []

        # Find columns tokenized_response_0 to tokenized_response_{n-1}
        for i in range(n_columns):
            column_name = f"tokenized_response_{i}"
            assert column_name in example, f"Column {column_name} not found in example"
            assert isinstance(
                example[column_name], list
            ), f"Column {column_name} is not a list in example {example}"
            all_words.extend(example[column_name])

        # Clean and process words
        for word in all_words:
            # cleaned_word = clean_word(word)
            if word:  # Skip empty/punctuation-only words
                # Update counts
                word_nationality_count[(word, nationality)] += 1
                word_count[word] += 1
                nationality_count[nationality] += 1
                total_word_count += 1

    print(f"Total words processed: {total_word_count}")
    print(f"Unique words: {len(word_count)}")
    print(f"Nationalities: {list(nationality_count.keys())}")

    # Calculate NPMI scores
    print("Calculating NPMI scores...")
    npmi_results = defaultdict(dict)

    # Progress bar for NPMI calculations
    for (word, nationality), co_occur_count in tqdm(
        word_nationality_count.items(),
        desc="Computing NPMI scores",
        unit="word-nationality pairs",
    ):
        # Calculate probabilities
        p_word_nationality = co_occur_count / total_word_count
        p_word = word_count[word] / total_word_count
        p_nationality = nationality_count[nationality] / total_word_count

        # Calculate PMI
        if p_word_nationality > 0 and p_word > 0 and p_nationality > 0:
            pmi = math.log(p_word_nationality / (p_word * p_nationality))

            # Calculate NPMI: PMI / (-log(P(word, nationality)))
            if p_word_nationality > 0:
                npmi = pmi / (-math.log(p_word_nationality))
                npmi_results[nationality][word] = npmi

    # Sort each nationality's words by NPMI score (descending)
    print("Sorting results by NPMI scores...")
    sorted_results = {}

    # Progress bar for sorting
    for nationality, word_scores in tqdm(
        npmi_results.items(), desc="Sorting nationality results", unit="nationalities"
    ):
        sorted_results[nationality] = dict(
            sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        )

    return sorted_results


def main():
    args = parse_args()

    # Load the dataset
    print(f"Loading dataset from {args.input_file}...")
    dataset = load_responses(args.input_file)

    # Calculate NPMI scores
    print(f"Processing {args.n} tokenized columns...")
    npmi_results = calculate_npmi(dataset, args.n)

    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(npmi_results, f, indent=2)

    print(f"NPMI calculation complete! Results saved to {args.output_file}")
    return npmi_results


if __name__ == "__main__":
    results = main()
