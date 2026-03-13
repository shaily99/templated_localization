import argparse
import json
import os
import sys

import editdistance as ed
import sacrebleu
from joblib import Parallel, delayed

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from tqdm import tqdm

from utils import load_responses


def parse_args():
    parser = argparse.ArgumentParser(description="Compute similarity within corpus.")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the corpus file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output file."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="wer",
        help="Similarity metric to use (e.g., wer).",
        choices=["wer", "bleu", "lcs"],
    )
    parser.add_argument(
        "--topic_keys_start_index",
        type=int,
        default=0,
        help="Start index for sorted topic keys (inclusive).",
    )
    parser.add_argument(
        "--topic_keys_slice_size",
        type=int,
        default=None,
        help="Slice size for sorted topic keys. If None, includes all topics from start index to the end.",
    )
    return parser.parse_args()


def lcs(responses1, responses2):
    def _lcs(a, b):
        if len(a) == 0 or len(b) == 0:
            return 1
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        best_len = 0
        best_end_i = 0  # end index in a

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > best_len:
                        best_len = dp[i][j]
                else:
                    dp[i][j] = 0
        lcs_value = 2 * best_len / (len(a) + len(b))
        return lcs_value

    scores = []
    for r1 in responses1:
        for r2 in responses2:
            lcs_value = _lcs(r1, r2)
            scores.append(lcs_value)
    return sum(scores) / len(scores) if scores else 0.0


def wer(responses1, responses2):
    def _wer(pred, ref):
        return (
            ed.eval(pred, ref) / max(len(pred), len(ref))
            if max(len(pred), len(ref)) > 0
            else 0.0
        )

    scores = []
    for r1 in responses1:
        for r2 in responses2:
            score1 = _wer(r1, r2)
            score2 = _wer(r2, r1)
            scores.append((score1 + score2) / 2)

    return sum(scores) / len(scores) if scores else 0.0


def bleu(responses1, responses2):
    def _bleu(pred, ref):
        bleu_score = sacrebleu.sentence_bleu(pred, [ref], tokenize="none")
        return bleu_score.score / 100.0

    scores = []
    for r1 in responses1:
        for r2 in responses2:
            score1 = _bleu(r1, r2)
            score2 = _bleu(r2, r1)
            scores.append((score1 + score2) / 2)

    return sum(scores) / len(scores) if scores else 0.0


def reformat_data(data, topic_keys):
    responses = {}
    print("Reformatting data.")
    for example in tqdm(data, total=len(data), desc="Reformatting: "):
        topic = example["topic"]
        if topic not in topic_keys:
            continue
        if topic not in responses:
            responses[topic] = {}
        nationality = example["nationality"]
        assert nationality not in responses[topic]
        responses[topic][nationality] = []
        for i in range(5):
            response_key = f"masked_response_{i}"
            responses[topic][nationality].append(example[response_key])
    assert len(responses) == len(topic_keys)
    return responses


def process_single_topic(topic, topic_data, nationalities, similarity_fn):
    """Process a single topic and return its results."""
    topic_result = {}

    for i in range(len(nationalities)):
        nationality1 = nationalities[i]
        for j in range(i + 1, len(nationalities)):
            nationality2 = nationalities[j]

            if nationality1 not in topic_result:
                topic_result[nationality1] = {}
            if nationality2 not in topic_result:
                topic_result[nationality2] = {}

            responses1 = topic_data[nationality1]
            responses2 = topic_data[nationality2]

            assert len(responses1) == len(responses2) == 5

            similarity_score = similarity_fn(responses1, responses2)
            topic_result[nationality1][nationality2] = similarity_score
            topic_result[nationality2][nationality1] = similarity_score

    return topic, topic_result


def main(data, nationalities, similarity_fn, topic_keys):
    reformatted_data = reformat_data(data, topic_keys)
    n_jobs = len(os.sched_getaffinity(0))
    delayed_runs = [
        delayed(process_single_topic)(
            topic=topic,
            topic_data=reformatted_data[topic],
            nationalities=nationalities,
            similarity_fn=similarity_fn,
        )
        for topic in topic_keys
    ]

    results_list = Parallel(
        n_jobs=n_jobs,
        backend="multiprocessing",
        verbose=1,
    )(
        tqdm(
            delayed_runs,
            desc=f"Processing topics with {n_jobs} cores: ",
            total=len(topic_keys),
        )
    )
    result = {}
    for topic, topic_result in results_list:
        result[topic] = topic_result
    return result


if __name__ == "__main__":
    args = parse_args()
    print(f"Arguments: {args}")
    data = load_responses(args.file)
    topic_keys = data.unique("topic")
    topic_keys = sorted(topic_keys)
    start_index = args.topic_keys_start_index
    slice_size = args.topic_keys_slice_size
    if slice_size is not None:
        end_index = start_index + slice_size
        if end_index > len(topic_keys):
            end_index = len(topic_keys)
    else:
        end_index = len(topic_keys)
    topic_keys = topic_keys[start_index:end_index]

    print(f"Start index (inclusive): {start_index}")
    print(f"End index (exclusive): {end_index}")

    output_path = args.output.replace(
        ".json", f"_start_index_{start_index}_end_index_{end_index}.json"
    )

    print(f"Number of topics to process: {len(topic_keys)}")
    # print(f"Topics to process: {topic_keys}")

    nationalities = list(set(data["nationality"]))
    nationalities = sorted(nationalities)
    if args.metric == "wer":
        similarity_fn = wer
    elif args.metric == "bleu":
        similarity_fn = bleu
    elif args.metric == "lcs":
        similarity_fn = lcs
    print(f"Output path: {output_path}")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping computation.")
        sys.exit(0)

    results = main(data, nationalities, similarity_fn, topic_keys)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
