import argparse
import os
import sys

import datasets

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from utils import load_responses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    print("Input directory: ", input_dir)
    print("Output file: ", output_file)
    min_idx = 50000
    max_idx = 0
    len_ctr = 0
    all_datasets = []
    for file in os.listdir(input_dir):
        start_idx = file.split("start_index_")[-1].split("_end_index")[0]
        end_idx = file.split("end_index_")[-1]
        if int(start_idx) < min_idx:
            min_idx = int(start_idx)
        if int(end_idx) > max_idx:
            max_idx = int(end_idx)
        data = load_responses(path=os.path.join(input_dir, file))
        print("File: ", file)
        len_ctr += len(data)
        all_datasets.append(data)
    print("Min index: ", min_idx)
    print("Max index: ", max_idx)
    assert min_idx == 0
    assert max_idx == 24125
    print("Total length: ", len_ctr)
    combined_dataset = datasets.concatenate_datasets(all_datasets)
    print("Combined dataset: ", combined_dataset)
    combined_dataset = combined_dataset.sort(
        ["source", "category", "topic", "nationality"]
    )
    print("Saved combined dataset to: ", output_file)
    print("Number of unique topics: ", len(combined_dataset.unique("topic")))
    print(
        "Number of unique nationalities: ", len(combined_dataset.unique("nationality"))
    )
    combined_dataset.save_to_disk(output_file)


if __name__ == "__main__":
    main()
