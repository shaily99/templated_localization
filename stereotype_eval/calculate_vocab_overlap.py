import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from utils import load_seegull_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seegull_path",
        type=str,
        default="/home/shailyjb/converging_cultural_narratives/stereotype_eval/dataset/stereotypes_global_v2.csv",
        help="Path to the Seegull dataset CSV file",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the JSON file containing NPMI sorted vocab per country",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file with overlap results",
    )
    return parser.parse_args()


def load_vocab_data(vocab_path):
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    for country in vocab_data:
        vocab_data[country] = list(vocab_data[country].keys())
    return vocab_data


def calculate_overlap(stereotypes, offensiveness, vocab_data):
    overlap_counts = {}
    overlap_percentages = {}
    offensiveness_scores = {}
    for country in stereotypes:
        if country not in vocab_data:
            continue
        overlap_counts[country] = {}
        overlap_percentages[country] = {}
        offensiveness_scores[country] = {}
        stereo_set = stereotypes[country]
        vocab = vocab_data[country]
        for percentage in range(1, 101, 1):
            mean_score = 0
            top_n = int(len(vocab) * (percentage / 100))
            top_vocab = set(list(vocab)[:top_n])
            overlap_count = 0
            total_count = len(stereo_set)
            for stereo in stereo_set:
                if stereo in top_vocab:
                    overlap_count += 1
                    mean_score += offensiveness[country].get(stereo, 0)
            overlap_counts[country][percentage] = overlap_count
            overlap_percentages[country][percentage] = (
                overlap_count / total_count if total_count > 0 else 0.0
            )
            if overlap_count > 0:
                mean_score /= overlap_count
            offensiveness_scores[country][percentage] = mean_score
    return overlap_counts, overlap_percentages, offensiveness_scores


if __name__ == "__main__":
    args = parse_args()
    (
        region_stereotypes,
        na_stereotypes,
        combined_stereotypes,
        region_stereotypes_majority,
        na_stereotypes_majority,
        combined_stereotypes_majority,
        offensiveness,
    ) = load_seegull_data(args.seegull_path)
    vocab_data = load_vocab_data(args.vocab_path)
    (
        region_stereotype_overlap_counts,
        region_stereotype_overlap_percentages,
        regional_offensiveness,
    ) = calculate_overlap(region_stereotypes, offensiveness, vocab_data)
    (
        na_stereotype_overlap_counts,
        na_stereotype_overlap_percentages,
        na_offensiveness,
    ) = calculate_overlap(na_stereotypes, offensiveness, vocab_data)
    (
        combined_stereotype_overlap_counts,
        combined_stereotype_overlap_percentages,
        combined_offensiveness,
    ) = calculate_overlap(combined_stereotypes, offensiveness, vocab_data)
    (
        region_stereotype_majority_overlap_counts,
        region_stereotype_majority_overlap_percentages,
        regional_majority_offensiveness,
    ) = calculate_overlap(region_stereotypes_majority, offensiveness, vocab_data)
    (
        na_stereotype_majority_overlap_counts,
        na_stereotype_majority_overlap_percentages,
        na_majority_offensiveness,
    ) = calculate_overlap(na_stereotypes_majority, offensiveness, vocab_data)
    (
        combined_stereotype_majority_overlap_counts,
        combined_stereotype_majority_overlap_percentages,
        combined_majority_offensiveness,
    ) = calculate_overlap(combined_stereotypes_majority, offensiveness, vocab_data)
    stereotype_data = {
        "regional": {
            "overlap_counts": region_stereotype_overlap_counts,
            "overlap_percentages": region_stereotype_overlap_percentages,
            "offensiveness_scores": regional_offensiveness,
        },
        "north_american": {
            "overlap_counts": na_stereotype_overlap_counts,
            "overlap_percentages": na_stereotype_overlap_percentages,
            "offensiveness_scores": na_offensiveness,
        },
        "combined": {
            "overlap_counts": combined_stereotype_overlap_counts,
            "overlap_percentages": combined_stereotype_overlap_percentages,
            "offensiveness_scores": combined_offensiveness,
        },
        "regional_majority": {
            "overlap_counts": region_stereotype_majority_overlap_counts,
            "overlap_percentages": region_stereotype_majority_overlap_percentages,
            "offensiveness_scores": regional_majority_offensiveness,
        },
        "north_american_majority": {
            "overlap_counts": na_stereotype_majority_overlap_counts,
            "overlap_percentages": na_stereotype_majority_overlap_percentages,
            "offensiveness_scores": na_majority_offensiveness,
        },
        "combined_majority": {
            "overlap_counts": combined_stereotype_majority_overlap_counts,
            "overlap_percentages": combined_stereotype_majority_overlap_percentages,
            "offensiveness_scores": combined_majority_offensiveness,
        },
    }

    with open(args.output_path, "w") as f:
        json.dump(stereotype_data, f, indent=4)
