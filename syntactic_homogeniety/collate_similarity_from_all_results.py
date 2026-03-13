import argparse
import json
import os
import pickle

from compute_similarity_save_all_results import load_results_pickle
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average scores across percentages."
    )
    parser.add_argument(
        "--percentages",
        type=str,
        required=True,
        help="Comma-separated list of percentages to include in the average calculation.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing score files for different masking percentages.",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="_start_index_0_end_index_125.pkl",
    )
    return parser.parse_args()


def create_file_path(base_dir, percentage, filename_suffix):
    path = f"{base_dir}/{percentage}{filename_suffix}"
    return path


def create_across_nationalities_story_pairs(nationalities):
    pairs = []
    for nat1 in range(len(nationalities)):
        for storyidx1 in range(5):
            for nat2 in range(nat1 + 1, len(nationalities)):
                for storyidx2 in range(5):
                    pair = (
                        (nationalities[nat1], storyidx1),
                        (nationalities[nat2], storyidx2),
                    )
                    pairs.append(pair)
    print(f"Created {len(pairs)} across-nationality story pairs.", flush=True)
    return pairs


def create_within_nationalities_story_pairs(nationalities):
    pairs = []
    for nat in range(len(nationalities)):
        for storyidx1 in range(5):
            for storyidx2 in range(storyidx1 + 1, 5):
                pair = (
                    (nationalities[nat], storyidx1),
                    (nationalities[nat], storyidx2),
                )
                pairs.append(pair)
    print(f"Created {len(pairs)} within-nationality story pairs.", flush=True)
    return pairs


def create_all_story_pairs(nationalities):
    all_pairs = create_across_nationalities_story_pairs(
        nationalities
    ) + create_within_nationalities_story_pairs(nationalities)
    print(f"Created {len(all_pairs)} total story pairs.", flush=True)
    return all_pairs


def calculate_avg(topic_scores, story_pairs):
    total = 0
    count = 0
    for (nat1, storyidx1), (nat2, storyidx2) in story_pairs:
        assert nat1 in topic_scores
        assert nat2 in topic_scores[nat1]
        assert storyidx1 < len(topic_scores[nat1][nat2])
        assert storyidx2 < len(topic_scores[nat1][nat2][storyidx1])
        score = topic_scores[nat1][nat2][storyidx1][storyidx2]
        total += score
        count += 1
    avg = total / count if count > 0 else 0
    return avg, total, count


# def create_across_nationality_pairs(nationalities):
#     pairs = []
#     for i in range(len(nationalities)):
#         for j in range(i + 1, len(nationalities)):
#             pairs.append((nationalities[i], nationalities[j]))
#     return pairs


# def get_pairs_across_nationalities(nationalities):
#     return create_across_nationality_pairs(nationalities)


# def get_pairs_within_nationalities(nationalities):
#     pairs = [(nat, nat) for nat in nationalities]
#     return pairs


# def get_all_pairs(nationalities):
#     pairs = get_pairs_across_nationalities(
#         nationalities
#     ) + get_pairs_within_nationalities(nationalities)
#     return pairs


# def calculate_avg_same_nationality_pair(scores):
#     assert len(scores) == 5 and len(scores[0]) == 5, f"Expected 5x5 scores"
#     total = 0
#     count = 0
#     # total 5 * 4 / 2 pairs expected
#     for i in range(len(scores)):
#         for j in range(i + 1, len(scores[i])):
#             total += scores[i][j]
#             count += 1
#     assert count == 10, f"Expected 10 pairs, got {count}"
#     avg = total / count
#     return avg, total, count


# def calculate_avg_different_nationality_pair(scores):
#     assert len(scores) == 5 and len(scores[0]) == 5, f"Expected 5x5 scores"
#     total = 0
#     count = 0
#     for i in range(len(scores)):
#         for j in range(len(scores[i])):
#             total += scores[i][j]
#             count += 1
#     assert count == 25, f"Expected 25 pairs, got {count}"
#     avg = total / count
#     return avg, total, count


# def calculate_avg(topic_scores, nationality_pairs):
#     overall_total = 0
#     overall_count = 0
#     for nat1, nat2 in nationality_pairs:
#         assert nat1 in topic_scores
#         assert nat2 in topic_scores[nat1]
#         if nat1 == nat2:
#             avg, total, count = calculate_avg_same_nationality_pair(
#                 topic_scores[nat1][nat2]
#             )
#         else:
#             avg, total, count = calculate_avg_different_nationality_pair(
#                 topic_scores[nat1][nat2]
#             )
#         overall_total += total
#         overall_count += count

#     overall_avg = overall_total / overall_count
#     return overall_avg, overall_total, overall_count


def init_results(topics):
    results = {}
    results["all_topics_macro"] = {
        "within_nationality": 0,
        "across_nationality": 0,
        "all_pairs": 0,
    }

    results["all_topics_micro"] = {
        "within_nationality": 0,
        "across_nationality": 0,
        "all_pairs": 0,
    }
    for topic in topics:
        results[topic] = {
            "within_nationality": {
                "total": 0,
                "count": 0,
                "avg": 0,
            },
            "across_nationality": {
                "total": 0,
                "count": 0,
                "avg": 0,
            },
            "all_pairs": {
                "total": 0,
                "count": 0,
                "avg": 0,
            },
        }

    return results


def calculate_overall_results(results):
    print("Calculating overall results.", flush=True)
    topics = [
        topic
        for topic in results.keys()
        if topic not in ["all_topics_macro", "all_topics_micro"]
    ]
    pair_types = list(results["all_topics_macro"].keys())
    for pair_type in pair_types:
        total = 0
        total_avg = 0
        count = 0
        for topic in topics:
            total += results[topic][pair_type]["total"]
            total_avg += results[topic][pair_type]["avg"]
            count += results[topic][pair_type]["count"]
        macro_avg = total_avg / len(topics)
        micro_avg = total / count
        results["all_topics_macro"][pair_type] = macro_avg
        results["all_topics_micro"][pair_type] = micro_avg
    return results


def process_percentage(scores):
    topics = list(scores.keys())
    nationalities = list(scores[topics[0]].keys())
    assert (
        len(nationalities) == 193 and len(topics) == 125
    ), f"Expected 193 nationalities and 125 topics, got {len(nationalities)} nationalities and {len(topics)} topics"

    results = init_results(topics)

    print("Getting nationality pairs.", flush=True)

    within_nationality_story_pairs = create_within_nationalities_story_pairs(
        nationalities
    )
    across_nationality_story_pairs = create_across_nationalities_story_pairs(
        nationalities
    )
    all_story_pairs = create_all_story_pairs(nationalities)

    for topic in tqdm(topics, desc="Processing topics: ", total=len(topics)):
        topic_scores = scores[topic]
        within_avg, within_total, within_count = calculate_avg(
            topic_scores, within_nationality_story_pairs
        )
        across_avg, across_total, across_count = calculate_avg(
            topic_scores, across_nationality_story_pairs
        )
        all_avg, all_total, all_count = calculate_avg(topic_scores, all_story_pairs)
        results[topic]["within_nationality"]["avg"] = within_avg
        results[topic]["within_nationality"]["total"] = within_total
        results[topic]["within_nationality"]["count"] = within_count
        results[topic]["across_nationality"]["avg"] = across_avg
        results[topic]["across_nationality"]["total"] = across_total
        results[topic]["across_nationality"]["count"] = across_count
        results[topic]["all_pairs"]["avg"] = all_avg
        results[topic]["all_pairs"]["total"] = all_total
        results[topic]["all_pairs"]["count"] = all_count

    results = calculate_overall_results(results)
    return results


if __name__ == "__main__":
    args = parse_args()
    percentages = args.percentages.split(",")
    paths = [
        create_file_path(args.base_dir, perc, args.filename_suffix)
        for perc in percentages
    ]

    all_results = {}
    # for path, percentage in tqdm(
    #     zip(paths, percentages), desc="Processing percentages: ", total=len(percentages)
    # ):
    for path, percentage in zip(paths, percentages):
        scores = load_results_pickle(path)
        print("Loaded scores for percentage:", percentage, flush=True)
        percentage_results = process_percentage(scores)
        all_results[percentage] = percentage_results
        del scores

    output_path = f"{args.base_dir}/combined_results_detailed_v2.json"
    print(f"Saving combined results to {output_path}", flush=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    all_results_small = {}
    for percentage in all_results:
        all_results_small[percentage] = {
            "all_topics_macro": all_results[percentage]["all_topics_macro"],
            "all_topics_micro": all_results[percentage]["all_topics_micro"],
        }
    output_path_small = f"{args.base_dir}/combined_results_macro_micro_v2.json"
    with open(output_path_small, "w") as f:
        json.dump(all_results_small, f, indent=4)

    print(f"Saved combined results to {output_path}", flush=True)
