# calculate average scores across percentages
import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average scores across percentages."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing score files for different masking percentages.",
    )
    return parser.parse_args()


def calculate_avg(scores):
    overall_avg = 0
    for topic in scores:
        avg = 0
        n = len(scores[topic])  # Length of identities
        for identity1 in scores[topic]:
            for identity2 in scores[topic][identity1]:
                if identity1 != identity2:
                    d = scores[topic][identity1][identity2]
                    avg += d
        overall_avg += avg / (n * (n - 1))
    return overall_avg / len(scores)


if __name__ == "__main__":
    args = parse_args()
    percentages = [str(i) for i in range(0, 100)]
    results = {}
    for perc in percentages:
        if "random" in args.input_dir:
            path = f"{args.input_dir}/{perc}_with_nationality_start_index_0_end_index_125.json"
        else:
            path = f"{args.input_dir}/{perc}_start_index_0_end_index_125.json"
        if not os.path.exists(path):
            # print(f"File not found: {path}, skipping.")
            continue
        scores = json.load(open(path, "r"))
        avg = calculate_avg(scores)
        results[perc] = avg
        print(f"Percentage: {perc}, AVG score: {avg}", flush=True)
    output_file = f"{args.input_dir}/avg_across_percentages.json"
    json.dump(results, open(output_file, "w"))
