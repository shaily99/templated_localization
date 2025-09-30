import argparse
import json
import os
import sys

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")


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
    print(f"Number of labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_args.get("learning_rate", 1e-3),
        per_device_train_batch_size=training_args.get("batch_size", 8),
        per_device_eval_batch_size=training_args.get("batch_size", 8),
        num_train_epochs=training_args.get("num_epochs", 5),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=training_args.get("seed", 47),
        save_total_limit=3,
        early_stopping_patience=training_args.get("early_stopping_patience", 2),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=utils.f1_score,
        callbacks=[EarlyStoppingCallback()],
    )

    print("Starting training...")

    trainer.train()
    print("Training completed.")

    print("Evaluating on validation set...")
    val_report = trainer.evaluate(eval_dataset=val_data)

    print("Evaluating on test set...")
    test_report = trainer.evaluate(eval_dataset=test_data)

    classifier_weights = trainer.model.classifier.weight.detach().cpu().numpy()

    return classifier_weights, val_report, test_report


if __name__ == "__main__":
    args = parse_args()

    output_dir = (
        f"{args.base_output_dir}/val_col_{args.val_column}_test_col_{args.test_column}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    print("Args:", args)
    print("Val column:", args.val_column)
    print("Test column:", args.test_column)
    print("Output dir:", output_dir)
    print("Model:", args.base_model)

    data = utils.load_responses(args.tokenized_responses)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    val_data = utils.create_data_split(
        data=data,
        column_indices=[args.val_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )
    test_data = utils.create_data_split(
        data=data,
        column_indices=[args.test_column],
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )
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

    classifier_weights, val_report, test_report = main(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_name=args.base_model,
        output_dir=output_dir,
    )

    print("\n--- Validation Report ---")
    print(val_report)

    print("\n--- Test Report ---")
    print(test_report)
