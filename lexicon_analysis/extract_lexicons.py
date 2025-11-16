import os
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set

from tqdm import tqdm
from datasets import load_from_disk


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_story_dict_from_dataset(
    dataset,
    response_prefix: str = "masked_response",
    identity_col: str = "nationality",
    allowed_identities: Optional[Set[str]] = None,
) -> Dict[str, List[List[str]]]:

    story_dict = defaultdict(list)

    for row in dataset:
        identity = row[identity_col]

        if allowed_identities is not None and identity not in allowed_identities:
            continue

        # Decide response index range
        if row.get(f"{response_prefix}_0") is None:
            start_idx = 1
            max_idx = 6  # 1..5
        else:
            start_idx = 0
            max_idx = 5  # 0..4

        for j in range(start_idx, max_idx):
            key = f"{response_prefix}_{j}"
            if key not in row:
                continue
            sent = row[key]
            if sent is None:
                continue
            if not isinstance(sent, list):
                continue
            story_dict[identity].append(sent)

    return story_dict


def get_lexicon_for_identity(
    masked_sents: List[List[str]],
    unmasked_sents: List[List[str]],
) -> List[str]:
    """Return list of unmasked tokens where masked token == [MASK]."""
    return [
        u_token
        for m_sent, u_sent in zip(masked_sents, unmasked_sents)
        for m_token, u_token in zip(m_sent, u_sent)
        if m_token == "[MASK]"
    ]


def sort_lexicons_by_npmi(
    lexicons: Dict[str, List[str]],
    npmi_data: Dict[str, Dict[str, float]],
) -> Dict[str, dict]:

    print("Sorting lexicons by NPMI scores...")
    output = {}

    for identity, words in lexicons.items():
        identity_npmi = npmi_data.get(identity, {})
        if not identity_npmi:
            continue

        word_freq = Counter(words)
        word_scores = {}

        for word, freq in word_freq.items():
            score = identity_npmi.get(word)
            if score is not None:
                word_scores[word] = (freq, score)

        if not word_scores:
            continue

        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1][1],  # sort by NPMI score
            reverse=True,
        )

        output[identity] = {
            "total_lexicons": len(sorted_words),
            "lexicons": {w: [freq, score] for w, (freq, score) in sorted_words},
        }

    print("Lexicons sorted.")
    return output


def find_mask_dir(base_path: str, threshold_val: float) -> str:

    candidates = []

    candidates.append(str(int(round(threshold_val))))

    if 0.0 <= threshold_val <= 1.0:
        candidates.append(str(int(round(threshold_val * 10))))   
        candidates.append(str(int(round(threshold_val * 100))))  

    candidates.append(str(threshold_val))
    candidates.append(f"{threshold_val:.2f}")

    tried = set()
    for c in candidates:
        if c in tried:
            continue
        tried.add(c)
        candidate_path = os.path.join(base_path, c)
        if os.path.isdir(candidate_path):
            return candidate_path

    return ""  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_base_path",
        type=str,
        required=True,
        help="Base dir with HF datasets for each masking level",
    )
    parser.add_argument(
        "--threshold_path",
        type=str,
        required=True,
        help="JSON with per-country masking breakpoint values",
    )
    parser.add_argument(
        "--npmi_path",
        type=str,
        required=True,
        help="JSON with NPMI scores: {country: {word: score}}",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to write lexicon JSON",
    )
    parser.add_argument(
        "--unmasked_dir_name",
        type=str,
        default="0",
        help="Subdir name under input_base_path containing unmasked stories",
    )
    parser.add_argument(
        "--response_prefix",
        type=str,
        default="masked_response",
        help="Prefix for response columns, default 'masked_response'",
    )
    parser.add_argument(
        "--identity_col",
        type=str,
        default="nationality",
        help="Column name for identity / country",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    thresholds = load_json(args.threshold_path)  
    npmi_data = load_json(args.npmi_path)        

    dir_to_identities: Dict[str, List[str]] = defaultdict(list)

    for identity, thr in thresholds.items():
        if thr is None:
            continue
        try:
            t_val = float(thr)
        except (TypeError, ValueError):
            print(f"Skipping {identity}: invalid threshold {thr}")
            continue

        mask_dir = find_mask_dir(args.input_base_path, t_val)
        if not mask_dir:
            print(f"Skipping {identity}: no dataset dir for threshold={t_val}")
            continue

        dir_to_identities[mask_dir].append(identity)

    print(f"Found {len(dir_to_identities)} masked directories to process.")

    # Load unmasked base dataset once
    unmasked_path = os.path.join(args.input_base_path, args.unmasked_dir_name)
    if not os.path.isdir(unmasked_path):
        print(f"ERROR: Unmasked dataset dir not found: {unmasked_path}")
        return

    print(f"Loading unmasked dataset from: {unmasked_path}")
    unmasked_ds = load_from_disk(unmasked_path)
    unmasked_story = build_story_dict_from_dataset(
        unmasked_ds,
        response_prefix=args.response_prefix,
        identity_col=args.identity_col,
    )
    print(f"Unmasked stories loaded for {len(unmasked_story)} identities.\n")

    # Per-identity lexicons
    lexicons: Dict[str, List[str]] = defaultdict(list)

    # Process each masked directory once, for the identities that use it
    for mask_dir, identities in tqdm(
        dir_to_identities.items(), desc="Processing masked dirs"
    ):
        try:
            ds = load_from_disk(mask_dir)
        except Exception as e:
            print(f"Failed to load dataset from {mask_dir}: {e}")
            continue

        needed_identities = set(identities)
        masked_story_all = build_story_dict_from_dataset(
            ds,
            response_prefix=args.response_prefix,
            identity_col=args.identity_col,
            allowed_identities=needed_identities,
        )

        for identity in identities:
            if identity not in masked_story_all:
                print(f"No masked data for {identity} in {mask_dir}")
                continue
            if identity not in unmasked_story:
                print(f"No unmasked data for {identity}, skipping")
                continue

            masked_sents = masked_story_all[identity]
            unmasked_sents = unmasked_story[identity]

            lexical_items = get_lexicon_for_identity(masked_sents, unmasked_sents)
            if lexical_items:
                lexicons[identity].extend(lexical_items)

    print(f"\nLexicons collected for {len(lexicons)} identities.\n")

    formatted_lexicons = sort_lexicons_by_npmi(lexicons, npmi_data)

    with open(args.output_path, "w") as f:
        json.dump(formatted_lexicons, f, indent=2)

    print("Lexicon extraction and saving complete.")


if __name__ == "__main__":
    main()
