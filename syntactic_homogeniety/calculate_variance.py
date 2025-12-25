# calculate lexical variance from wer scores for different percentages of syntactic masking

import argparse
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate lexical variance from WER scores for different syntactic masking percentages."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing WER score files for different masking percentages.",
    )
    return parser.parse_args()


def calculate_variance_(wer_scores):
    overall_var = 0
    for topic in wer_scores:
        var = 0
        n = len(wer_scores[topic])  # Length of identities
        for identity1 in wer_scores[topic]:
            for identity2 in wer_scores[topic][identity1]:
                if identity1 != identity2:
                    d = wer_scores[topic][identity1][identity2]
                    var += d**2
        overall_var += var / (n**2)
    return overall_var / len(wer_scores)


if __name__ == "__main__":
    args = parse_args()
    percentages = [str(i) for i in range(0, 100)]
    results = {}
    for perc in percentages:
        if "random" in args.input_dir:
            path = f"{args.input_dir}/{perc}_with_nationality_start_index_0_end_index_125.json"
        else:
            path = f"{args.input_dir}/{perc}_start_index_0_end_index_125.json"
        wer_scores = json.load(open(path, "r"))
        variance = calculate_variance_(wer_scores)
        results[perc] = variance
        print(f"Percentage: {perc}, Variance: {variance}", flush=True)
    output_file = f"{args.input_dir}/variance_across_percentages.json"
    json.dump(results, open(output_file, "w"))
