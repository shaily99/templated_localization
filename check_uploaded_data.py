import argparse

import datasets

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument(
        "--hf_repo_name",
        type=str,
        default="shaily99/madlibs",
        help="Hugging Face repository name",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        required=True,
        help="Subset name for the dataset. Usually, the model name.",
    )
    parser.add_argument(
        "--split_name", type=str, default="temp_0.7", help="Split name to check"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = datasets.load_dataset(
        args.hf_repo_name,
        args.subset_name,
        download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD,
    )
    print(len(data["temp_0.7"]))
    assert len(data["temp_0.7"]) == 24125
