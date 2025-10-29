#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
import json
from datasets import load_from_disk
sys.path.insert(0, str(Path(__file__).parent))
from embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def sanitize(n: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in n)

def cache_path(nat: str, cache_dir: Path) -> Path:
    return cache_dir / f"{sanitize(nat)}.npz"

def main():
    ap = argparse.ArgumentParser(description="Precompute caches for a shard of nationalities in one process.")
    ap.add_argument("--shard_file", required=True, help="File with one nationality per line.")
    ap.add_argument("--data_path", required=True, help="HF dataset directory (load_from_disk).")
    ap.add_argument("--tokenization_path", required=True, help="Precomputed NPMI tokenization JSON.")
    ap.add_argument("--cache_dir", default="cache_npmis", help="Output dir for <nat>.npz.")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--n_columns", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # read shard list
    nats = [l.strip() for l in Path(args.shard_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not nats:
        logging.error(f"No nationalities found in {args.shard_file}")
        sys.exit(1)

    tok = json.loads(Path(args.tokenization_path).read_text(encoding="utf-8"))
    logging.info(f"Tokenization JSON: {len(tok)} nationalities")
    for n in nats:
        if n not in tok:
            logging.warning(f"{n} not in tokenization JSON; will likely skip with empty result.")

    # load dataset 
    ds = load_from_disk(args.data_path)
    story_nats = {ex["nationality"] for ex in ds}
    keep = []
    for n in nats:
        if n in story_nats:
            keep.append(n)
        else:
            logging.warning(f"{n} not found in dataset; skipping.")
    nats = keep
    if not nats:
        logging.error("No valid nationalities remain after dataset/tokenization checks.")
        sys.exit(1)

    # init model/tokenizer 
    eg = EmbeddingGenerator(model_name=args.model_name, tokenization_path=args.tokenization_path)

    # process each nationality
    for nat in nats:
        out_npz = cache_path(nat, cache_dir)
        if out_npz.exists() and not args.force:
            logging.info(f"[SKIP] cache exists for {nat}: {out_npz}")
            continue

        # targeted story build to avoid all-nat pass
        story_dict = eg.prepare_story_data_for(ds, nat, n_columns=args.n_columns)
        if nat not in story_dict or len(story_dict[nat]) == 0:
            logging.warning(f"No stories for {nat}; skipping.")
            continue

        words, embs = eg.generate_nationality_embeddings(
            story_dict, nat, batch_size=args.batch_size, cache_dir=args.cache_dir
        )
        logging.info(f"[OK] {nat}: cached {len(words)} words -> {out_npz}")

if __name__ == "__main__":
    main()
