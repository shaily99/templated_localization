import argparse
import json
import logging
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
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the evaluation report.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
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


def run_inference(model, test_data, batch_size, output_dir):
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/temp_eval",
        per_device_eval_batch_size=batch_size,
        logging_dir=f"{output_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_data,
        compute_metrics=utils.compute_f1_score,
    )

    logger = logging.getLogger(__name__)
    logger.info("Running evaluation...")
    # eval_results = trainer.evaluate()
    detailed_report, metrics_df = utils.create_report(trainer, test_data)

    # return eval_results, detailed_report, metrics_df
    return detailed_report, metrics_df


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/training.log"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{'='*60}")
    logger.info("DEVICE INFORMATION")
    logger.info(f"{'='*60}")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        logger.info("No GPU available, using CPU")

    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"{'='*60}\n")

    logger.info(f"Args: {args}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data = utils.load_responses(args.tokenized_responses)
    # data = data.select(range(min(50, len(data))))  # for quick sanity check

    logger.info("Creating test split...")

    test_data = utils.create_data_split(
        data=data,
        column_indices=[args.test_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    logger.info(f"Test samples: {len(test_data)}")
    num_labels = len(set(test_data["label"]))
    logger.info(f"Number of labels: {num_labels}")

    # eval_results
    detailed_report = run_inference(
        model=model,
        test_data=test_data,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    logger.info(f"Eval results: {detailed_report}")
    # eval_results_path = os.path.join(output_dir, "eval_results.json")
    # with open(eval_results_path, "w") as f:
    #     json.dump(eval_results, f, indent=2, default=float)
    detailed_report_path = os.path.join(output_dir, "detailed_report.json")
    with open(detailed_report_path, "w") as f:
        json.dump(detailed_report, f, indent=2, default=float)
    logger.info(f"Detailed report saved to {detailed_report_path}")
    # metrics_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    # metrics_df.to_csv(metrics_csv_path, index=False)
    # logger.info(f"Confusion matrix saved to {metrics_csv_path}")
    logger.info("Evaluation complete.")
