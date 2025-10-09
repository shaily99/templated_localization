import argparse
import json
import logging
import os
import random
import sys

import torch
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

HOME = os.path.expanduser("~")

sys.path.insert(0, f"{HOME}/converging_cultural_narratives")

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length. Default set based on max length of default model.",
    )
    parser.add_argument(
        "--tokenized_responses",
        type=str,
        required=True,
        help="Path to the tokenized responses file.",
    )
    parser.add_argument(
        "--val_column",
        type=int,
        required=True,
        help="Column index for validation data.",
    )
    parser.add_argument(
        "--test_column",
        type=int,
        required=True,
        help="Column index for test data.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses per prompt.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and val and test report.",
    )
    parser.add_argument(
        "--seed", type=int, default=47, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="Learning rate for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="culture-bias-mmbert",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional run name for wandb",
    )
    return parser.parse_args()


def main(
    train_data,
    val_data,
    test_data,
    model_name,
    output_dir,
    training_args={},
):

    num_labels = len(set(train_data["label"]))
    logger.info(f"Number of labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args_obj = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_args.get("learning_rate", 3e-5),
        per_device_train_batch_size=training_args.get("batch_size", 8),
        per_device_eval_batch_size=training_args.get("batch_size", 8),
        num_train_epochs=training_args.get("num_epochs", 5),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1_weighted",
        seed=training_args.get("seed", 47),
        save_total_limit=3,
        fp16=training_args.get("fp16", False),
        gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 1),
        report_to=["wandb"],
        run_name=training_args.get("wandb_run_name"),
        max_grad_norm=1.0,
        warmup_steps=training_args.get("warmup_steps", 500),
    )

    trainer = Trainer(
        model=model,
        args=training_args_obj,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=utils.compute_f1_score,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.get("early_stopping_patience", 2)
            )
        ],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    logger.info("Evaluating on validation set...")
    val_report = trainer.evaluate(eval_dataset=val_data)

    logger.info("Evaluating on test set...")
    test_report = trainer.evaluate(eval_dataset=test_data)

    return val_report, test_report, trainer


if __name__ == "__main__":
    args = parse_args()
    output_dir = (
        f"{args.base_output_dir}/val_col_{args.val_column}_test_col_{args.test_column}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

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
    logger.info(f"Val column: {args.val_column}")
    logger.info(f"Test column: {args.test_column}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model: {args.base_model}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "base_model": args.base_model,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "fp16": args.fp16,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "val_column": args.val_column,
            "test_column": args.test_column,
        },
    )

    data = utils.load_responses(args.tokenized_responses)

    # # for testing
    # random_indices = random.sample(range(len(data)), 3000)
    # data = [data[i] for i in random_indices]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    logger.info("Creating val split...")

    val_data = utils.create_data_split(
        data=data,
        column_indices=[args.val_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    logger.info("Creating test split...")

    test_data = utils.create_data_split(
        data=data,
        column_indices=[args.test_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    logger.info("Creating train split...")

    train_column_indices = [
        i
        for i in range(args.num_responses)
        if i != args.val_column and i != args.test_column
    ]
    assert len(train_column_indices) == args.num_responses - 2
    train_data = utils.create_data_split(
        data=data,
        column_indices=train_column_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")

    logger.info("Validating label mapping...")
    all_train_labels = set(train_data["label"])
    for label_idx in all_train_labels:
        nationality = utils.get_nationality_of_index(int(label_idx))
        if nationality == "NOT FOUND":
            logger.warning(
                f"Warning: Label index {label_idx} has no nationality mapping!"
            )

    num_labels = len(set(train_data["label"]))
    logger.info(f"Number of labels: {num_labels}")

    training_args_dict = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "fp16": args.fp16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "wandb_run_name": args.wandb_run_name,
    }

    val_report, test_report, trainer = main(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_name=args.base_model,
        output_dir=output_dir,
        training_args=training_args_dict,
    )

    logger.info("\n--- Validation Report (from Trainer) ---")
    logger.info(val_report)

    logger.info("\n--- Test Report (from Trainer) ---")
    logger.info(test_report)

    # Generate detailed reports with per-class metrics
    logger.info("\n--- Generating Detailed Reports ---")

    val_detailed, val_metrics_df = utils.create_report(trainer, val_data)
    test_detailed, test_metrics_df = utils.create_report(trainer, test_data)

    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION PER-CLASS F1 SCORES")
    logger.info(f"{'='*60}")
    for nationality, metrics in sorted(
        val_detailed["per_class_metrics"].items(),
        key=lambda x: x[1]["f1-score"],
        reverse=True,
    ):
        logger.info(
            f"{nationality:20s}: F1={metrics['f1-score']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Support={int(metrics['support'])}"
        )

    logger.info(f"\n{'='*60}")
    logger.info("TEST PER-CLASS F1 SCORES")
    logger.info(f"{'='*60}")
    for nationality, metrics in sorted(
        test_detailed["per_class_metrics"].items(),
        key=lambda x: x[1]["f1-score"],
        reverse=True,
    ):
        logger.info(
            f"{nationality:20s}: F1={metrics['f1-score']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Support={int(metrics['support'])}"
        )

    # Save model and tokenizer
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    logger.info(f"Best model saved to {output_dir}/best_model")

    with open(f"{output_dir}/val_report.json", "w") as f:
        json.dump(val_report, f, indent=2, default=float)
    with open(f"{output_dir}/test_report.json", "w") as f:
        json.dump(test_report, f, indent=2, default=float)

    metadata = {
        "base_model": args.base_model,
        "num_labels": num_labels,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "best_val_f1": val_report.get("eval_f1", None),
        "test_f1": test_report.get("eval_f1", None),
        "hyperparameters": training_args_dict,
        "train_columns": train_column_indices,
        "val_column": args.val_column,
        "test_column": args.test_column,
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Validation F1: {val_report.get('eval_f1', 0):.4f}")
    logger.info(f"Overall Test F1: {test_report.get('eval_f1', 0):.4f}")
    logger.info(f"\nAll results saved to {output_dir}")
    logger.info(f"  - validation_detailed_report.json")
    logger.info(f"  - validation_per_class_metrics.csv")
    logger.info(f"  - test_detailed_report.json")
    logger.info(f"  - test_per_class_metrics.csv")
    logger.info(f"  - best_model/")

    wandb.finish()
    wandb.finish()
