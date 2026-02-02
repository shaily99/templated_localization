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
        "--community_counts_path",
        type=str,
        required=True,
        help="Path to the JSON file containing NPMI sorted vocab per country",
    )
    parser.add_argument(
        "--npmi_path",
        type=str,
        required=True,
        help="Path to the JSON file containing NPMI scores per country",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file with overlap results",
    )
    return parser.parse_args()


def precision(stereotypes, counts, npmi):
    precision = {}
    tf_precision = {}

    for country in counts:
        if country not in stereotypes:
            continue
        precision[country] = {}
        tf_precision[country] = {}
        for percentage in range(1, 101, 1):
            all_words = list(npmi[country].keys())
            top_n = int(len(all_words) * (percentage / 100))
            top_vocab = set(all_words[:top_n])
            stereo_set = stereotypes[country]
            overlap_count = 0
            tf_overlap_count = 0
            total_count = 0
            tf_total_count = 0
            for word in top_vocab:
                if word in stereo_set:
                    overlap_count += 1
                    tf_overlap_count += counts[country][word]
                total_count += 1
                tf_total_count += counts[country][word]
            precision[country][percentage] = (
                overlap_count / total_count if total_count > 0 else 0
            )
            tf_precision[country][percentage] = (
                tf_overlap_count / tf_total_count if tf_total_count > 0 else 0
            )
    return precision, tf_precision


if __name__ == "__main__":
    args = parse_args()
    (
        region_stereotypes,
        na_stereotypes,
        combined_stereotypes,
        region_stereotypes_majority,
        na_stereotypes_majority,
        combined_stereotypes_majority,
        _,
    ) = load_seegull_data(args.seegull_path)
    with open(args.community_counts_path, "r") as f:
        community_counts = json.load(f)
    with open(args.npmi_path, "r") as f:
        npmi_scores = json.load(f)
    region_precision, region_tf_precision = precision(
        region_stereotypes, community_counts, npmi_scores
    )
    na_precision, na_tf_precision = precision(
        na_stereotypes, community_counts, npmi_scores
    )
    combined_precision, combined_tf_precision = precision(
        combined_stereotypes, community_counts, npmi_scores
    )
    region_majority_precision, region_majority_tf_precision = precision(
        region_stereotypes_majority, community_counts, npmi_scores
    )
    na_majority_precision, na_majority_tf_precision = precision(
        na_stereotypes_majority, community_counts, npmi_scores
    )
    combined_majority_precision, combined_majority_tf_precision = precision(
        combined_stereotypes_majority, community_counts, npmi_scores
    )
    output = {
        "region_precision": region_precision,
        "region_tf_precision": region_tf_precision,
        "na_precision": na_precision,
        "na_tf_precision": na_tf_precision,
        "combined_precision": combined_precision,
        "combined_tf_precision": combined_tf_precision,
        "region_majority_precision": region_majority_precision,
        "region_majority_tf_precision": region_majority_tf_precision,
        "na_majority_precision": na_majority_precision,
        "na_majority_tf_precision": na_majority_tf_precision,
        "combined_majority_precision": combined_majority_precision,
        "combined_majority_tf_precision": combined_majority_tf_precision,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Precision results saved to {args.output_path}")
