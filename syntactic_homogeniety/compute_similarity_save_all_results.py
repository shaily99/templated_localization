import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np
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
        default="lcs",
        help="Similarity metric to use (e.g., lcs).",
        choices=["lcs", "shingle", "shingle_freq_weighted"],
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


def make_scratch_dir(output_path):
    key = output_path.replace(".json", "").replace("/", "_").replace(".", "_")
    scratch_dir = f"/scratch/shailyjb_{key}"
    # assert not os.path.exists(
    #     scratch_dir
    # ), f"Scratch directory {scratch_dir} already exists."
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


def clean_scratch_dir(scratch_dir):
    print(f"Cleaning up scratch directory {scratch_dir}")
    for filename in os.listdir(scratch_dir):
        file_path = os.path.join(scratch_dir, filename)
        os.remove(file_path)
    os.rmdir(scratch_dir)


def save_results_pickle(results, output_path):
    pickle_path = output_path.replace(".json", ".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved pickle to {pickle_path}")


def load_results_pickle(output_path):
    with open(output_path, "rb") as f:
        return pickle.load(f)


def lcs(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best_len = max(best_len, dp[i][j])
            else:
                dp[i][j] = 0
    return 2 * best_len / (m + n)


def create_ngrams(token_list, n=4):
    ngrams = {}
    for i in range(len(token_list) - n + 1):
        ngram = token_list[i : i + n]
        ngrams[tuple(ngram)] = ngrams.get(tuple(ngram), 0) + 1
    return ngrams


def shingle_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a_ngrams = a.keys()
    b_ngrams = b.keys()
    intersection = len(set(a_ngrams) & set(b_ngrams))
    union = len(set(a_ngrams) | set(b_ngrams))
    return intersection / union


def shingle_similarity_freq_weighted(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    a_ngrams = a.keys()
    b_ngrams = b.keys()
    all_ngrams = set(a_ngrams) | set(b_ngrams)

    intersection = sum(min(a.get(ng, 0), b.get(ng, 0)) for ng in all_ngrams)
    union = sum(max(a.get(ng, 0), b.get(ng, 0)) for ng in all_ngrams)

    return intersection / union


def reformat_data(data, topic_keys, metric="lcs"):
    responses = {}
    print("Reformatting data.")
    for example in tqdm(data, total=len(data), desc="Reformatting: "):
        topic = example["topic"]
        if topic not in topic_keys:
            continue
        if topic not in responses:
            responses[topic] = {}
        nationality = example["nationality"]
        if nationality not in responses[topic]:
            responses[topic][nationality] = {}
        else:
            print(
                f"Warning: Duplicate entry for topic '{topic}' and {nationality} in the data. Overwriting previous entry."
            )
        for i in range(5):
            response_key = f"masked_response_{i}"
            if metric == "lcs":
                responses[topic][nationality][i] = example[response_key]
            elif metric == "shingle" or metric == "shingle_freq_weighted":
                responses[topic][nationality][i] = create_ngrams(example[response_key])
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return responses


def process_single_topic(topic, topic_data, nationalities, similarity_fn, scratch_dir):
    topic_result = {
        n1: {n2: [[None] * 5 for _ in range(5)] for n2 in nationalities}
        for n1 in nationalities
    }

    for idx1, nationality1 in enumerate(nationalities):
        for idx2 in range(idx1, len(nationalities)):
            nationality2 = nationalities[idx2]

            for i in range(5):
                for j in range(5):
                    if nationality1 == nationality2 and i == j:
                        score = 1.0
                    else:
                        r1 = topic_data[nationality1][i]
                        r2 = topic_data[nationality2][j]
                        score = similarity_fn(r1, r2)

                    topic_result[nationality1][nationality2][i][j] = score
                    topic_result[nationality2][nationality1][j][i] = score

    path = os.path.join(scratch_dir, f"{topic}.pkl")
    save_results_pickle({topic: topic_result}, path)
    return topic, path


def main(reformatted_data, nationalities, similarity_fn, topic_keys, scratch_dir):
    n_jobs = len(os.sched_getaffinity(0))
    delayed_runs = [
        delayed(process_single_topic)(
            topic=topic,
            topic_data=reformatted_data[topic],
            nationalities=nationalities,
            similarity_fn=similarity_fn,
            scratch_dir=scratch_dir,
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

    print("Aggregating results.", flush=True)

    result = {}
    for topic, path in tqdm(
        results_list, desc="Aggregating results: ", total=len(results_list)
    ):
        result.update(load_results_pickle(path))
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

    nationalities = list(set(data["nationality"]))
    nationalities = sorted(nationalities)
    # if args.metric == "wer":
    #     similarity_fn = wer
    # elif args.metric == "bleu":
    #     similarity_fn = bleu
    if args.metric == "lcs":
        similarity_fn = lcs
    elif args.metric == "shingle":
        similarity_fn = shingle_similarity
    elif args.metric == "shingle_freq_weighted":
        similarity_fn = shingle_similarity_freq_weighted
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

    print(f"Output path: {output_path}")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping computation.")
        sys.exit(0)

    scratch_dir = make_scratch_dir(output_path)

    reformatted_data = reformat_data(data, topic_keys, metric=args.metric)

    results = main(
        reformatted_data, nationalities, similarity_fn, topic_keys, scratch_dir
    )
    # print(results, flush=True)
    print(f"Saving results to {output_path}")
    save_results_pickle(results, output_path)
    clean_scratch_dir(scratch_dir)
