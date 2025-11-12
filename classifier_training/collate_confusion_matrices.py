import argparse
import json

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collate confusion matrices from different validation and test splits."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--classifier_dir_add",
        type=str,
        default="",
        options=["", "_with_nationality"],
        help="Suffix to add to the directory names.",
    )
    parser.add_argument(
        "--template_add",
        type=str,
        default="",
        options=["", "_with_nationality"],
        help="Additional string to add to the template name.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=47,
        help="Random seed used in the directory names.",
    )
    return parser.parse_args()


vals = [0, 1, 2, 3, 4]
tests = [1, 2, 3, 4, 0]


if __name__ == "__main__":
    args = parse_args()
    base_dir = f"/data/group_data/starlight/culture_bias/classifiers/0.7/mmBERT-base/compressed_masked_responses/{args.model}"

    classifier_dir_add = args.classifier_dir_add
    template_add = args.template_add
    random_name = f"random_{args.random_seed}_"

    percentages = [f"0{template_add}"] + [str(i) for i in range(100)]

    cms = {}
    for percentage in tqdm(percentages):
        for val, test in zip(vals, tests):
            dir = f"{base_dir}/val_{val}_test_{test}{classifier_dir_add}/{random_name}test_reports"
            cm_path = f"{dir}/{percentage}{template_add}.json"
            report = json.load(open(cm_path, "r"))
            cm = pd.DataFrame(report["confusion_matrix"]["matrix"])
            cm.columns = report["confusion_matrix"]["labels"]
            columns = [f"predicted_{label}" for label in cm.columns]
            cm.columns = columns
            cm.index = report["confusion_matrix"]["labels"]
            indices = [f"true_{label}" for label in cm.index]
            cm.index = indices
            if percentage not in cms:
                cms[percentage] = cm
            else:
                cms[percentage] += cm
        cms[percentage] = cms[percentage].T.to_dict()

    output_path = (
        f"{base_dir}/{random_name}collated_confusion_matrix{classifier_dir_add}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2, ensure_ascii=False)

    print(f"Collated confusion matrices saved to {output_path}")
