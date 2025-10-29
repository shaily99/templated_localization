import json
import re
import string

import datasets
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from nationality_country import get_index_of_nationality, get_nationality_of_index


def load_responses(path):
    data = datasets.load_from_disk(path)
    print(f"Path: {path}")
    print(f"Loaded dataset: {data}")
    return data


def tokenize_text(text: str) -> list:
    """Tokenize text using NLTK word_tokenize."""
    if text is None or text == "":
        return []

    def replace_unicode_punct(text):
        # Replace common Unicode punctuation with ASCII equivalents
        replacements = {
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u2015": "-",  # horizontal bar
            "\u2018": "'",  # left single quotation
            "\u2019": "'",  # right single quotation
            "\u201c": '"',  # left double quotation
            "\u201d": '"',  # right double quotation
            "\u2026": ".",  # horizontal ellipsis
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        return text

    def add_spaces_around_punctuation(text):
        # Add spaces around punctuation
        punctuation_pattern = f"([{re.escape(string.punctuation)}])"
        result = re.sub(punctuation_pattern, r" \1 ", text)
        result = re.sub(r"\s+", " ", result)
        return result.strip()

    text = add_spaces_around_punctuation(replace_unicode_punct(text))
    tokens = word_tokenize(text)
    tokens = [token.lower().strip() for token in tokens if token.lower().strip()]
    return tokens


def create_data_split(data, column_indices, tokenizer, max_length=8192, seed=47):
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    inputs = []
    labels = []
    for index in column_indices:
        assert (
            f"masked_response_{index}" in data[0].keys()
        ), f"Column index {index} not found in data."
        for example in data:
            input_text = (
                " ".join(example[f"masked_response_{index}"])
                .replace("[MASK]", "[UNK]")
                .strip()
            )
            label = get_index_of_nationality(example["nationality"])
            inputs.append(input_text)
            labels.append(label)
    split = {"text": inputs, "label": labels}
    split = datasets.Dataset.from_dict(split)
    split.shuffle(seed=seed)
    split = split.map(tokenize, batched=True)

    print("Number of indices used:", len(column_indices))
    print("Length of split:", len(split))
    return split


def compute_f1_score(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_per_class = f1_score(labels, predictions, average=None)

    metrics = {
        "avg_f1_weighted": f1_weighted,
        "avg_f1_micro": f1_micro,
        "avg_f1_macro": f1_macro,
    }
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        nationality = get_nationality_of_index(int(label))
        metrics[f"f1_{nationality}"] = f1_per_class[i]

    return metrics


def create_report(trainer, dataset):
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = np.argmax(logits, axis=-1)

    all_labels = sorted(np.unique(np.concatenate([labels, predictions])))
    label_names = [get_nationality_of_index(int(idx)) for idx in all_labels]

    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_macro = f1_score(labels, predictions, average="macro")

    report_dict = classification_report(
        labels,
        predictions,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, predictions, labels=all_labels)

    cm_with_labels = {
        "matrix": cm.tolist(),
        "labels": label_names,
        "label_indices": all_labels,
    }

    detailed_report = {
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "classification_report": report_dict,
        "confusion_matrix": cm_with_labels,
    }

    return detailed_report
