#!/usr/bin/env python3
import itertools
import json
from datasets import load_from_disk

TOK_PATH = "/data/group_data/starlight/culture_bias/homomorphs/tokenized_npmi/gpt_4o_mini_tokenized.json"
DATA_PATH = "/data/group_data/starlight/culture_bias/data/model_outputs/0.7/tokenized_responses/gpt_4o_mini"

with open(TOK_PATH) as f:
    tok_data = json.load(f)
dataset = load_from_disk(DATA_PATH)
story_nats = {ex['nationality'] for ex in dataset}

print(f"Found {len(story_nats)} nationalities")

with open("nationalities.txt", "w") as f:
    for nat in story_nats:
        f.write(f"{nat}\n")

print("Saved to nationalities.txt")

# Generate all pairs
pairs = list(itertools.combinations(story_nats, 2))
print(f"Total pairs: {len(pairs)}")

with open('nationality_pairs.txt', 'w') as f:
    for nat1, nat2 in pairs:
        f.write(f"{nat1}\t{nat2}\n")

print("Saved to nationality_pairs.txt")