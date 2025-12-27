import argparse
import json

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seegull_path",
        type=str,
        default="/home/shailyjb/converging_cultural_narratives/stereotype_analysis/dataset/stereotypes_global_v2.csv",
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


def load_seegull_data(seegull_path):
    df = pd.read_csv(seegull_path)
    region_stereotypes = {}
    na_stereotypes = {}
    offensiveness_scores = {}
    for _, row in df.iterrows():
        country = row["identity"]
        stereotype = row["attribute"].strip().lower()
        region_stereo = int(row["region_stereo"])
        na_stereo = int(row["NA_stereo"])
        offensiveness = float(row["mean offensiveness_score"])
        if country not in region_stereotypes:
            region_stereotypes[country] = set()
            na_stereotypes[country] = set()
            offensiveness_scores[country] = {}
        if region_stereo > 0:
            region_stereotypes[country].add(stereotype)
        if na_stereo > 0:
            na_stereotypes[country].add(stereotype)
        offensiveness_scores[country][stereotype] = offensiveness
    return region_stereotypes, na_stereotypes, offensiveness_scores


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
        overlap_counts[country] = {}
        overlap_percentages[country] = {}
        offensiveness_scores[country] = {}
        stereo_set = stereotypes[country]
        vocab = vocab_data.get(country, [])
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
    region_stereotypes, na_stereotypes, offensiveness = load_seegull_data(
        args.seegull_path
    )
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
    }

    with open(args.output_path, "w") as f:
        json.dump(stereotype_data, f, indent=4)
