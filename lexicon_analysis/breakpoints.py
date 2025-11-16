import json
import argparse
import os
from glob import glob
from kneed import KneeLocator

def parse_args():
    parser = argparse.ArgumentParser(description="Extract masking thresholds per region")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory containing a single collated metrics JSON file",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["random", "floor", "elbow", "min_random_floor"],
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.02,
        help="F1 threshold for signal floor (used with --mode floor)",
    )
    return parser.parse_args()

def load_all_folds(input_path):
    if os.path.isdir(input_path):
        json_files = sorted(glob(os.path.join(input_path, "*.json")))
    else:
        json_files = [input_path]

    all_folds = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            all_folds.append(json.load(f))
    return all_folds

def load_and_average_f1_scores(all_folds):
    first = all_folds[0]

    # Sort masking levels numerically, but keep original string keys
    sorted_keys = sorted(first.keys(), key=float)
    thresholds = [float(k) for k in sorted_keys]

    # Use first threshold to get list of regions
    sample_per_class = first[sorted_keys[0]]["per_class"]
    regions = list(sample_per_class.keys())

    region_scores = {region: [] for region in regions}

    for k in sorted_keys:
        for region in regions:
            fold_f1s = []
            for fold in all_folds:
                # fold[k]["per_class"][region]["f1"]
                per_class = fold[k]["per_class"]
                metrics = per_class.get(region)
                if metrics is None:
                    f1 = 0.0
                else:
                    f1 = metrics.get("f1", 0.0)
                fold_f1s.append(f1)
            avg_f1 = sum(fold_f1s) / len(fold_f1s)
            region_scores[region].append(avg_f1)

    return thresholds, region_scores

def find_breaking_point_random(thresholds, scores, num_classes):
    random_f1 = 1 / num_classes
    results = {}
    for region, f1_list in scores.items():
        for t, f1 in zip(thresholds, f1_list):
            if f1 <= random_f1:
                results[region] = t
                break
        else:
            results[region] = None
    return results

def find_breaking_point_floor(thresholds, scores, floor=0.02):
    results = {}
    for region, f1_list in scores.items():
        for i in range(len(thresholds)):
            if f1_list[i] < floor and all(f < floor for f in f1_list[i:]):
                results[region] = thresholds[i]
                break
        else:
            results[region] = None
    return results

def find_elbow_point(thresholds, scores):
    results = {}
    for region, f1_list in scores.items():
        try:
            # thresholds is a list of floats, sorted ascending
            kneedle = KneeLocator(thresholds, f1_list, curve="convex", direction="decreasing")
            results[region] = kneedle.knee
        except Exception:
            results[region] = None
    return results

def find_min_random_floor(thresholds, scores, num_classes, floor=0.02):
    random_results = find_breaking_point_random(thresholds, scores, num_classes)
    floor_results = find_breaking_point_floor(thresholds, scores, floor)
    combined_results = {}
    for region in scores.keys():
        rand_val = random_results.get(region)
        floor_val = floor_results.get(region)
        vals = [v for v in (rand_val, floor_val) if v is not None]
        combined_results[region] = min(vals) if vals else None
    return combined_results

def main():
    args = parse_args()
    all_folds = load_all_folds(args.input_path)
    thresholds, region_scores = load_and_average_f1_scores(all_folds)

    if args.mode == "random":
        results = find_breaking_point_random(thresholds, region_scores, args.num_classes)
    elif args.mode == "floor":
        results = find_breaking_point_floor(thresholds, region_scores, args.floor)
    elif args.mode == "elbow":
        results = find_elbow_point(thresholds, region_scores)
    elif args.mode == "min_random_floor":
        results = find_min_random_floor(thresholds, region_scores, args.num_classes, args.floor)

    with open(args.output_path, "w") as f_out:
        json.dump(results, f_out, indent=2)

if __name__ == "__main__":
    main()
