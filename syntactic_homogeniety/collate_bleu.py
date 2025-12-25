# calculate average BLEU across percentages

import argparse
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average BLEU across percentages."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing BLEU score files for different masking percentages.",
    )
    return parser.parse_args()


def calculate_avg_bleu(bleu_scores):
    overall_avg = 0
    for topic in bleu_scores:
        avg = 0
        n = len(bleu_scores[topic])  # Length of identities
        for identity1 in bleu_scores[topic]:
            for identity2 in bleu_scores[topic][identity1]:
                if identity1 != identity2:
                    d = bleu_scores[topic][identity1][identity2]
                    avg += d
        overall_avg += avg / (n * (n - 1))
    return overall_avg / len(bleu_scores)


if __name__ == "__main__":
    args = parse_args()
    percentages = [str(i) for i in range(0, 100)]
    results = {}
    for perc in percentages:
        if "random" in args.input_dir:
            path = f"{args.input_dir}/{perc}_with_nationality_start_index_0_end_index_125.json"
        else:
            path = f"{args.input_dir}/{perc}_start_index_0_end_index_125.json"
        bleu_scores = json.load(open(path, "r"))
        avg = calculate_avg_bleu(bleu_scores)
        results[perc] = avg
        print(f"Percentage: {perc}, AVG BLEU: {avg}", flush=True)
    output_file = f"{args.input_dir}/avg_across_percentages.json"
    json.dump(results, open(output_file, "w"))
