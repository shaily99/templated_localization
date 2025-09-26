import argparse
import json
import sys

import editdistance as ed

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
        choices=["wer", "bleu"],
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


def wer(responses1, responses2):
    def _wer(pred, ref):
        return ed.eval(pred, ref) / max(len(pred), len(ref))

    scores = []
    for r1 in responses1:
        for r2 in responses2:
            if r1 == r2:
                continue
            score1 = _wer(r1, r2)
            score2 = _wer(r2, r1)
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
        responses[topic][nationality] = []
        for i in range(5):
            response_key = f"masked_response_{i}"
            responses[topic][nationality].append(example[response_key])
    assert len(responses) == len(topic_keys)
    return responses


def main(data, nationalities, similarity_fn, topic_keys):
    reformatted_data = reformat_data(data, topic_keys)
    result = {}
    for topic in tqdm(reformatted_data, total=len(reformatted_data), desc="Topic: "):
        if topic not in result:
            result[topic] = {}
        for i in tqdm(
            range(len(nationalities)), total=len(nationalities), desc="Nationality: "
        ):
            nationality1 = nationalities[i]
            for j in range(i, len(nationalities)):
                nationality2 = nationalities[j]
                if nationality1 not in result[topic]:
                    result[topic][nationality1] = {}
                if nationality2 not in result[topic]:
                    result[topic][nationality2] = {}

                responses1 = reformatted_data[topic][nationality1]
                responses2 = reformatted_data[topic][nationality2]

                assert len(responses1) == len(responses2) == 5

                similarity_score = similarity_fn(responses1, responses2)
                result[topic][nationality1][nationality2] = similarity_score
                result[topic][nationality2][nationality1] = similarity_score
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
    print(f"Topics to process: {topic_keys}")

    # data = data.filter(lambda x: x["topic"] in topic_keys)
    nationalities = list(set(data["nationality"]))
    nationalities = sorted(nationalities)
    if args.metric == "wer":
        similarity_fn = wer
    elif args.metric == "bleu":
        # similarity_fn = bleu
        raise NotImplementedError("BLEU metric not implemented yet.")

    results = main(data, nationalities, similarity_fn, topic_keys)
    print(f"Output path: {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
