import argparse
import pandas as pd
import ast
import json
from collections import defaultdict, Counter
from typing import List, Dict
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Compute overlap between SeeGull and either stories or lexicons")
    parser.add_argument("--stories_path", type=str, required=True, help="TSV file for stories or JSON for lexicons")
    parser.add_argument("--stereotype_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    return parser.parse_args()


def load_stereotype_dict(stereo_df, threshold=2):
    with open("seegull-mapping.json") as f:
        identity_map = json.load(f)
    # Reverse the mapping: from mapped value → original label(s)
    reversed_map = {}
    for k, v_list in identity_map.items():
        for v in v_list:
            reversed_map[v] = k

    stereo_dict = defaultdict(list)
    for _, row in stereo_df.iterrows():
        if row["region_stereo"] >= threshold:
            raw_identity = row["identity"]
            mapped_to = reversed_map.get(raw_identity, raw_identity)
            stereo_dict[mapped_to].append(row["attribute"])
    print("Loaded stereotype dataset...")

    return stereo_dict


def analyze_token_overlap(identity: str, tokens: List[str], stereotype_dict: Dict[str, List[str]]):
    stereo_words = set(stereotype_dict.get(identity, []))
    token_counts = Counter(tokens)
    token_set = set(tokens)
    global_union_size = len(token_set | stereo_words)
    overlap = stereo_words & token_counts.keys()
    return {
        "overlap_len": len(overlap),
        "overlap_ratio": len(overlap) / len(set(tokens)) if tokens else 0.0,
        "total_lexicons": len(set(tokens)),
        "seegull_lexicons": len(stereo_words),
        "overlap_words": {w: token_counts[w] for w in overlap},
        "jaccard_index": len(overlap) / global_union_size if global_union_size else 0.0
    }


def process_lexicon_file(path, stereotype_dict):
    print("Processing lexicon file...")
    with open(path) as f:
        lexicon_data = json.load(f)

    results = []
    identity_overlap_sets = defaultdict(set)

    for identity, data in lexicon_data.items():
        lexicons = data.get("lexicons", {})
        tokens = list(lexicons.keys())
        counts = {w: lexicons[w][0] for w in tokens}

        result = analyze_token_overlap(identity, tokens, stereotype_dict)
        result["total_lexicons"] = len(tokens)
        result["identity"] = identity
        result["overlap_words"] = {w: counts[w] for w in result["overlap_words"]}
        identity_overlap_sets[identity].update(result["overlap_words"].keys())
        results.append(result)

    return results, identity_overlap_sets


def main():
    args = parse_args()
    stereo_df = pd.read_csv(args.stereotype_data_path)
    stereotype_dict = load_stereotype_dict(stereo_df, threshold=args.threshold)

    overlap_results, overlap_sets = process_lexicon_file(args.stories_path, stereotype_dict)

    # Save .tsv output
    output_df = pd.DataFrame(overlap_results)
    output_df.to_csv(args.output_path, sep="\t", index=False, quoting=csv.QUOTE_ALL, quotechar='"')

    # Save JSON cumulative overlap
    overlap_json_path = args.output_path.replace(".tsv", "_cumulative_overlap.json")
    with open(overlap_json_path, "w") as f:
        json.dump({k: sorted(list(v)) for k, v in overlap_sets.items()}, f, indent=2)


if __name__ == "__main__":
    main()
