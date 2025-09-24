# code to upload data to HF

import argparse

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument(
        "--local_data_path", type=str, required=True, help="Path to the local dataset"
    )
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
        "--split_name", type=str, default="temp_0.7", help="Split name to upload."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = utils.load_responses(
        args.local_data_path,
    )
    # # for testing, upload 100 data points
    # data = data.select(range(100))
    data.push_to_hub(
        args.hf_repo_name,
        private=True,
        config_name=args.subset_name,
        split=args.split_name,
    )
