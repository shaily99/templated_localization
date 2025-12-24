import argparse
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        "Calculate metrics from collated confusion matrices."
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
        choices=["", "_with_nationality"],
        help="Suffix to add to the directory names.",
    )
    parser.add_argument(
        "--template_add",
        type=str,
        default="",
        choices=["", "_with_nationality"],
        help="Additional string to add to the template name.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=-1,
        help="Random seed used in the directory names.",
    )
    return parser.parse_args()


def confusion_matrix_to_metrics(confusion_dict):

    classes = [key.replace("true_", "") for key in confusion_dict.keys()]
    classes = list(sorted(classes))

    total_samples = sum(
        sum(predictions.values()) for predictions in confusion_dict.values()
    )
    assert len(classes) == 193
    assert total_samples == 120625
    # print(f"Classes: {classes}")
    # print(f"Total samples: {total_samples}")

    metrics = {}

    for cls in classes:
        true_key = f"true_{cls}"
        pred_key = f"predicted_{cls}"

        tp = confusion_dict[true_key][pred_key]
        fn = sum(confusion_dict[true_key].values()) - tp
        fp = sum(
            confusion_dict[other_true][pred_key]
            for other_true in confusion_dict.keys()
            if other_true != true_key
        )
        tn = total_samples - tp - fp - fn

        support = tp + fn
        assert support == 625

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[cls] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    return metrics


def calculate_overall_metrics(class_metrics):
    classes = list(class_metrics.keys())
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(classes)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(classes)

    total_support = sum(m["support"] for m in class_metrics.values())
    assert total_support == 120625
    weighted_precision = (
        sum(m["precision"] * m["support"] for m in class_metrics.values())
        / total_support
        if total_support > 0
        else 0.0
    )
    weighted_recall = (
        sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )

    total_tp = sum(m["tp"] for m in class_metrics.values())
    total_fp = sum(m["fp"] for m in class_metrics.values())
    total_fn = sum(m["fn"] for m in class_metrics.values())

    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    overall_accuracy = total_tp / total_support if total_support > 0 else 0.0

    return {
        "total_support": total_support,
        "accuracy": round(overall_accuracy, 4),
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1": round(weighted_f1, 4),
        },
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
    }


if __name__ == "__main__":

    percentages = [str(i) for i in range(100)]

    args = parse_args()
    print(args)
    model = args.model
    classifier_dir_add = args.classifier_dir_add
    template_add = args.template_add
    random_name = f"random_{args.random_seed}_" if args.random_seed > 0 else ""

    base_dir = f"/data/group_data/starlight/culture_bias/classifiers/0.7/mmBERT-base/compressed_masked_responses/{model}"
    cm_path = f"{base_dir}/{random_name}collated_confusion_matrix_classifier{classifier_dir_add}_template{template_add}.json"
    op_path = f"{base_dir}/{random_name}collated_metrics_classifier{classifier_dir_add}_template{template_add}.json"

    cms = json.load(open(cm_path, "r"))
    metrics = {}
    for percentage in tqdm(percentages):
        if percentage not in cms:
            p = percentage + "_with_nationality"
            if p in cms:
                cm = cms[p]
            else:
                raise ValueError(f"Percentage {percentage} not found in cms.")
        else:
            cm = cms[percentage]
        per_class_metrics = confusion_matrix_to_metrics(cm)
        overall_metrics = calculate_overall_metrics(per_class_metrics)
        metrics[percentage] = {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
        }

    with open(op_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Collated metrics saved to {op_path}")
