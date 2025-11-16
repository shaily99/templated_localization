import argparse
import ast
import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
import pyarrow as pa
from datasets import Dataset
from datasets import load_from_disk
from tqdm import tqdm


class TransformationPipeline:
    def __init__(
        self,
        original_stories_path: str,
        masked_stories_dir: str,
        homomorphs_dir: str,
        breakpoints_file: str,
    ):
        self.original_ds = load_from_disk(original_stories_path)
        self.masked_dir = Path(masked_stories_dir)
        self.hom_dir = Path(homomorphs_dir)

        with open(breakpoints_file) as f:
            self.breakpoints = json.load(f)

    def _load_masked_for_source(self, source: str, threshold: int):
        masked_path = self.masked_dir / f"{int(threshold)}"
        masked_ds = load_from_disk(str(masked_path))
        return masked_ds.filter(lambda x: x["nationality"] == source)

    def _load_original_for_source(self, source: str):
        return self.original_ds.filter(lambda x: x["nationality"] == source)

    def _load_hom_map(self, source: str, target: str):
        hom_path = self.hom_dir / f"{source}_{target}.csv"
        if not hom_path.exists():
            return None

        df = pd.read_csv(hom_path)
        hom_map = {}
        for _, row in df.iterrows():
            hom_list = ast.literal_eval(row["homomorphs"])
            if hom_list:
                hom_map[row["word"]] = hom_list[0]
        return hom_map

    def transform_source(
        self,
        source: str,
        target: Optional[str] = None,
    ) -> pd.DataFrame:

        if source not in self.breakpoints:
            return pd.DataFrame()

        threshold = int(self.breakpoints[source])

        masked_stories = self._load_masked_for_source(source, threshold)
        original_stories = self._load_original_for_source(source)

        n_rows = min(len(original_stories), len(masked_stories))
        if n_rows == 0:
            return pd.DataFrame()

        targets = (
            [target]
            if target
            else [t for t in self.breakpoints.keys() if t != source]
        )

        rows = []

        for tgt in tqdm(targets, desc=f"Targets({source})"):

            hom_map = self._load_hom_map(source, tgt)
            if hom_map is None:
                continue

            for idx in tqdm(
                range(n_rows),
                desc=f"{source}->{tgt}",
                leave=False,
            ):
                orig_story = original_stories[idx]
                masked_story = masked_stories[idx]

                row = {
                    "source_nationality": source,
                    "target_nationality": tgt,
                    "threshold": threshold,
                    "category": orig_story["category"],
                    "topic": orig_story["topic"],
                }

                for fold in range(5):
                    orig_key = f"tokenized_response_{fold}"
                    mask_key = f"masked_response_{fold}"

                    original_tokens = orig_story.get(orig_key)
                    masked_tokens = masked_story.get(mask_key)

                    if not original_tokens or not masked_tokens:
                        row[f"original_response_{fold}"] = None
                        row[f"template_response_{fold}"] = None
                        row[f"transformed_response_{fold}"] = None
                        continue

                    # fill masks
                    filled = []
                    for i, tok in enumerate(masked_tokens):
                        if tok == "[MASK]":
                            src_tok = original_tokens[i] if i < len(original_tokens) else tok
                            filled.append(hom_map.get(src_tok, src_tok))
                        else:
                            filled.append(tok)

                    row[f"original_response_{fold}"] = " ".join(original_tokens)
                    row[f"template_response_{fold}"] = " ".join(masked_tokens)
                    row[f"transformed_response_{fold}"] = " ".join(filled)

                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()


    def save_for_source(
    self,
    source: str,
    output_dir: str,
    target: Optional[str] = None,):
        
        out_dir = Path(output_dir) / source
        out_dir.mkdir(parents=True, exist_ok=True)

        df = self.transform_source(source, target=target)
        if df.empty:
            print(f"[INFO] No output for {source}; nothing saved.")
            return

        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.save_to_disk(str(out_dir))
        print(f"[OK] Saved {len(ds)} rows to HF dataset directory: {out_dir}")

    def save_many(
        self,
        sources: List[str],
        output_dir: str,
        target: Optional[str] = None,
    ):
        for source in tqdm(sources, desc="Sources"):
            self.save_for_source(source, output_dir, target=target)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original_stories_path", required=True)
    p.add_argument("--masked_stories_dir", required=True)
    p.add_argument("--homomorphs_dir", required=True)
    p.add_argument("--breakpoints_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source")
    p.add_argument("--sources_file")
    p.add_argument("--target")
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = TransformationPipeline(
        original_stories_path=args.original_stories_path,
        masked_stories_dir=args.masked_stories_dir,
        homomorphs_dir=args.homomorphs_dir,
        breakpoints_file=args.breakpoints_file,
    )

    if args.source:
        sources = [args.source]
    elif args.sources_file:
        with open(args.sources_file) as f:
            sources = [x.strip() for x in f if x.strip()]
    else:
        sources = list(pipeline.breakpoints.keys())

    if args.test:
        df = pipeline.transform_source(sources[0], target=args.target)
        print(df.head())
    else:
        pipeline.save_many(sources, args.output_dir, target=args.target)


if __name__ == "__main__":
    main()
