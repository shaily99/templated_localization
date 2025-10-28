import argparse
import json
import os
import sys

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

HOME = os.path.expanduser("~")
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, f"{HOME}/converging_cultural_narratives")

import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a test dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--tokenized_responses",
        type=str,
        required=True,
        help="Path to the tokenized responses file.",
    )
    parser.add_argument(
        "--detailed_report_path",
        type=str,
        required=True,
        help="Path to save the evaluation report.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--test_column",
        type=int,
        required=True,
        help="Column index for test data.",
    )
    parser.add_argument(
        "--seed", type=int, default=47, help="Random seed for reproducibility."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("No GPU available, using CPU")

    print(f"PyTorch Version: {torch.__version__}")

    print(f"Args: {args}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data = utils.load_responses(args.tokenized_responses)
    # data = data.select(range(min(50, len(data))))  # for quick sanity check

    print("Creating test split...")

    test_data = utils.create_data_split(
        data=data,
        column_indices=[args.test_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(f"Test samples: {len(test_data)}")
    num_labels = len(set(test_data["label"]))
    print(f"Number of labels: {num_labels}")

    trainer = Trainer(
        model=model,
    )

    print("Running evaluation...")
    detailed_report = utils.create_report(trainer, test_data)

    with open(args.detailed_report_path, "w") as f:
        json.dump(detailed_report, f, indent=2, default=float)
    print(f"Detailed report saved to {args.detailed_report_path}")
    print("Evaluation complete.")
