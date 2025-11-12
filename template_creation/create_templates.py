import argparse
import json
import os
import random
import string
import sys

from nltk.tokenize import word_tokenize

HOME = os.path.expanduser("~")

sys.path.insert(0, f"{HOME}/converging_cultural_narratives")

from nationality_country import nationality_to_country
from utils import load_responses


def parse_args():
    parser = argparse.ArgumentParser(description="Create templates")
    parser.add_argument(
        "--tokenized_input_file",
        help="Path to input dataset file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_file", help="Path to output file", required=True, type=str
    )
    parser.add_argument(
        "--n",
        help="Number of tokenized columns to process (0 to n-1)",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--npmi_file", help="Path to npmi file", required=True, type=str
    )
    parser.add_argument(
        "--percentage_threshold",
        help="Percentage threshold for templates",
        default=0.0,
        type=int,
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Whether to compress the masked responses by removing consecutive mask tokens",
    )
    parser.add_argument("--randomize", action="store_true", default=False)
    parser.add_argument(
        "--random_seed", help="Random seed. Used only if randomize is True.", default=47
    )
    parser.add_argument("--mask_token", help="Mask token", default="[MASK]", type=str)
    args = parser.parse_args()
    return args


def get_masking_words(npmi, percentage_threshold, randomize=False, random_seed=47):
    masking_words = {}
    for nationality, word_npmi in npmi.items():
        words = list(word_npmi.keys())
        num_words = len(words)
        num_words_to_mask = int((num_words * percentage_threshold) / 100)
        if not randomize:
            words_to_mask = words[:num_words_to_mask]
        else:
            random.seed(random_seed)
            words_to_mask = random.sample(words, num_words_to_mask)
        masking_words[nationality] = words_to_mask
    return masking_words


def mask_country_nationality(masked_response, nationality, mask_token="[MASK]"):
    countries = nationality_to_country[nationality].split(",")
    countries_tokenized = []
    for country in countries:
        country = country.lower().strip()
        country_tokens = word_tokenize(country)
        if country_tokens:  # Only add non-empty token lists
            countries_tokenized.append(country_tokens)

    nationality_tokens = word_tokenize(nationality.lower())

    # Collect all token sequences to search for (countries + nationality)
    sequences_to_mask = countries_tokenized + [nationality_tokens]

    # Sort sequences by length (longest first) to prioritize longer matches
    sequences_to_mask.sort(key=len, reverse=True)

    # print("sequences_to_mask: ", sequences_to_mask)

    # Work with lowercase version for matching
    masked_response_lower = []
    for token in masked_response:
        if token == mask_token:
            masked_response_lower.append(mask_token)
        else:
            masked_response_lower.append(token.lower())
    result_response = masked_response_lower.copy()

    # For each sequence, find and replace in the response
    for sequence in sequences_to_mask:
        if len(sequence) == 0:
            continue

        # Find all occurrences of this sequence
        i = 0
        while i <= len(masked_response_lower) - len(sequence):
            # Check if sequence matches at position i
            match = True
            for j, token in enumerate(sequence):
                if (
                    i + j >= len(masked_response_lower)
                    or masked_response_lower[i + j] != token
                ):
                    match = False
                    break

            if match:
                # Replace the entire sequence with a single mask token
                # Remove the sequence length tokens and insert one mask token
                for _ in range(len(sequence)):
                    result_response.pop(i)
                    masked_response_lower.pop(i)

                result_response.insert(i, mask_token)
                masked_response_lower.insert(i, mask_token)
                i += 1  # Move past the inserted mask token
            else:
                i += 1

    return result_response


def compress_masked_response(response):
    compressed_response = []
    previous_word = None
    for word in response:
        if word == "[MASK]":
            if previous_word != "[MASK]":
                compressed_response.append(word)
        else:
            compressed_response.append(word)
        previous_word = word
    return compressed_response


def mask_response(response, masking_words, mask_token="[MASK]"):
    masked_response = []
    for word in response:
        word = word.lower() if word != mask_token else word
        if word in string.punctuation:
            masked_response.append(word)
        else:
            if word in masking_words:
                masked_response.append(mask_token)
            else:
                masked_response.append(word)

    return masked_response


def create_templates(
    tokenized_data,
    masking_words,
    n,
    compress=False,
    mask_token="[MASK]",
):
    def process_example(example):
        nationality = example["nationality"]
        for i in range(n):
            masked_response = example[f"tokenized_response_{i}"]
            # First mask the country and nationalities
            masked_response = mask_country_nationality(
                masked_response=masked_response,
                nationality=nationality,
                mask_token=mask_token,
            )
            # Then mask the words based on NPMI
            masked_response = mask_response(
                response=masked_response,
                masking_words=masking_words[nationality],
                mask_token=mask_token,
            )

            # Finally compress if needed
            if compress:
                masked_response = compress_masked_response(masked_response)

            example[f"masked_response_{i}"] = masked_response

        return example

    return tokenized_data.map(process_example)


def drop_tokenized_responses(tokenized_data, n):
    columns_to_remove = [f"tokenized_response_{i}" for i in range(n)]
    return tokenized_data.remove_columns(columns_to_remove)


def main():
    args = parse_args()
    print(f"Args: {args}")
    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists. Exiting.")
        return
    tokenized_data = load_responses(args.tokenized_input_file)
    npmi = json.load(open(args.npmi_file))
    # print(f"Num countries in npmi: {len(npmi)}")
    # print(f"Num countries in COUNTRIES: {len(COUNTRIES)}")
    # print("NPMIs: ", npmi.keys())
    # print("COUNTRIES: ", COUNTRIES)
    # assert len(npmi) == len(COUNTRIES)
    masking_words = get_masking_words(
        npmi=npmi,
        percentage_threshold=args.percentage_threshold,
        randomize=args.randomize,
        random_seed=args.random_seed,
    )
    # print(f"Masking words: {masking_words}")
    masked_data = create_templates(
        tokenized_data=tokenized_data,
        masking_words=masking_words,
        n=args.n,
        compress=args.compress,
        mask_token=args.mask_token,
    )
    masked_data = drop_tokenized_responses(masked_data, args.n)
    print(f"Masked data: {masked_data}")
    del tokenized_data
    masked_data.save_to_disk(args.output_file)
    print(f"Saved masked data to {args.output_file}")
    print(masked_data["masked_response_0"][0])


if __name__ == "__main__":
    main()
