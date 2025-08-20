import argparse
import re
import sys

from nltk.tokenize import word_tokenize

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from utils import load_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input HuggingFace dataset directory",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the tokenized dataset"
    )

    parser.add_argument(
        "--n",
        type=int,
        help="Number of response columns (0 to n-1) to tokenize",
        default=5,
    )

    return parser.parse_args()


def replace_unicode_punct(text):
    # Replace common Unicode punctuation with ASCII equivalents
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
        "\u2015": "---",  # horizontal bar
        "\u2018": "'",  # left single quotation
        "\u2019": "'",  # right single quotation
        "\u201c": '"',  # left double quotation
        "\u201d": '"',  # right double quotation
        "\u2026": "...",  # horizontal ellipsis
    }

    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    return text


def tokenize_text(text: str) -> list:
    """Tokenize text using NLTK word_tokenize."""
    if text is None or text == "":
        return []
    text = replace_unicode_punct(text)
    return word_tokenize(text)


def process_dataset(dataset, n: int):
    """Process the dataset to tokenize response columns."""
    print(f"Processing dataset with {len(dataset)} rows")
    print(f"Tokenizing response columns 0 to {n-1}")
    print(f"Original columns: {dataset.column_names}")

    def tokenize_responses(example):
        """Tokenize all response columns for a single example."""
        for i in range(n):
            response_col = f"response_{i}"
            tokenized_col = f"tokenized_response_{i}"
            assert response_col in example
            tokens = tokenize_text(example[response_col])
            example[tokenized_col] = tokens
        return example

    # Apply tokenization to all examples
    tokenized_dataset = dataset.map(tokenize_responses)

    # Drop original response columns
    columns_to_remove = [f"response_{i}" for i in range(n)]
    assert len(columns_to_remove) == n
    if columns_to_remove:
        print(f"Removing original response columns: {columns_to_remove}")
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    print(f"Final columns: {tokenized_dataset.column_names}")
    tokenized_dataset = tokenized_dataset.sort(
        ["source", "category", "topic", "nationality"]
    )
    return tokenized_dataset


def main():
    args = parse_args()

    print(f"Input dataset: {args.input}")
    print(f"Output path: {args.output}")
    print(f"Number of response columns: {args.n}")

    # Load the HuggingFace dataset
    print("Loading dataset...")
    dataset = load_responses(args.input)
    print(f"Loaded dataset with {len(dataset)} examples")

    # Process the dataset
    print("Starting tokenization...")
    tokenized_dataset = process_dataset(dataset, args.n)

    # Save the processed dataset
    print(f"Saving tokenized dataset to: {args.output}")
    tokenized_dataset.save_to_disk(args.output)
    print("Tokenization complete!")


if __name__ == "__main__":
    main()
