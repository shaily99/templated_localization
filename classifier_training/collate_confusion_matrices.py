import json

import pandas as pd
from tqdm import tqdm

model = "llama_3-3_70b_instruct"
add = ""

vals = [0, 1, 2, 3, 4]
tests = [1, 2, 3, 4, 0]

percentages = ["0_with_nationality"] + [str(i) for i in range(100)]
base_dir = f"/data/group_data/starlight/culture_bias/classifiers/0.7/mmBERT-base/compressed_masked_responses/{model}"

cms = {}
for percentage in tqdm(percentages):
    for val, test in zip(vals, tests):
        dir = f"{base_dir}/val_{val}_test_{test}{add}/test_reports"
        cm_path = f"{dir}/{percentage}.json"
        report = json.load(open(cm_path, "r"))
        cm = pd.DataFrame(report["confusion_matrix"]["matrix"])
        cm.columns = report["confusion_matrix"]["labels"]
        cm.index = report["confusion_matrix"]["labels"]
        if percentage not in cms:
            cms[percentage] = cm
        else:
            cms[percentage] += cm
    cms[percentage] = cms[percentage].to_dict()

output_path = f"{base_dir}/collated_confusion_matrix{add}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cms, f, indent=2, ensure_ascii=False)

print(f"Collated confusion matrices saved to {output_path}")
