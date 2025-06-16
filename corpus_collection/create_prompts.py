import sys

sys.path.append("..")

import argparse
import csv
from typing import Dict, List, Tuple

import pandas as pd

from io_utils import read_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create prompts from templates, nationalities, and topics"
    )
    parser.add_argument(
        "--templates",
        help="Path to templates file",
        default="data/templates.tsv",
    )
    parser.add_argument(
        "--nationalities",
        help="Path to nationalities file",
        default="data/nationalities.tsv",
    )
    parser.add_argument(
        "--topics",
        help="Path to topics file",
        default="data/topics.tsv",
    )
    parser.add_argument(
        "--output",
        help="Path to output file",
        default="data/prompts.tsv",
    )

    return parser.parse_args()


def handle_vowels(template: str, nationality: str) -> str:
    """Handles vowels preceding identity terms in templates."""
    if "{vowel}" not in template:
        return template

    if nationality[0].lower() in "aeiou":
        template = template.replace("{vowel}", "an")
    else:
        template = template.replace("{vowel}", "a")
    return template


def generate_prompts(
    templates: Dict[str, str],
    nationalities: List[str],
    topics: Dict[str, List[Tuple[str, str]]],
) -> List[Dict[str, str]]:
    """Generate prompts by combining templates, nationalities, and topics."""
    prompts: List[Dict[str, str]] = []

    for category, topic_data in topics.items():
        if category not in templates:
            raise ValueError(f"Category {category} not found in templates.")

        template = templates[category]

        for item in topic_data:
            source, topic = item
            for nationality in nationalities:
                prompt = template.replace("{nationality}", nationality)
                prompt = prompt.replace("{topic}", topic)
                prompt = handle_vowels(prompt, nationality)

                prompts.append(
                    {
                        "source": source,
                        "category": category,
                        "topic": topic,
                        "nationality": nationality,
                        "template": template,
                        "prompt": prompt,
                    }
                )

    return prompts


def main() -> None:
    """Main function to orchestrate prompt generation."""
    args = parse_args()

    print(f"Templates file: {args.templates}")
    print(f"Nationalities file: {args.nationalities}")
    print(f"Topics file: {args.topics}")

    print("Reading input files...")
    # Read and validate input files
    templates_df = read_file(args.templates)
    nationalities_df = read_file(args.nationalities)
    topics_df = read_file(args.topics)

    # Process data
    templates = templates_df.set_index("category")["template"].to_dict()
    nationalities = nationalities_df["nationality"].dropna().tolist()
    topics = (
        topics_df.groupby("category")
        .apply(lambda x: list(zip(x["source"], x["concept"])))
        .to_dict()
    )

    print(f"Templates loaded: {len(templates)}")
    print(f"Nationalities loaded: {len(nationalities)}")
    print(f"Topics loaded: {len(topics)} categories")
    for category, topic_list in topics.items():
        print(f"Number of {category} topics: {len(topic_list)}")

    # Generate prompts
    print("Generating prompts...")
    prompts = generate_prompts(templates, nationalities, topics)

    print(f"Length of generated prompts: {len(prompts)}")

    # Write output
    print(f"Output file: {args.output}")
    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv(
        args.output, sep="\t", index=False, quoting=csv.QUOTE_ALL, quotechar='"'
    )
    print(f"Successfully wrote {len(prompts)} prompts to: {args.output}")


if __name__ == "__main__":
    main()
