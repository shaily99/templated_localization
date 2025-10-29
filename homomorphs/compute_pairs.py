#!/usr/bin/env python3
import argparse, logging
import numpy as np
from pathlib import Path
from homomorphs import HomomorphismExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def sanitize(n: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in n)

def load_nat(cache_dir: str, nat: str):
    f = Path(cache_dir) / (sanitize(nat) + ".npz")
    data = np.load(f, allow_pickle=True)
    types = data["words"].astype(object).tolist()
    embs = data["embs"].astype(np.float32)
    if len(types) == 0 or embs.size == 0:
        raise ValueError(f"Empty cache for {nat}: {f}")
    return types, embs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat1", required=True)
    ap.add_argument("--nat2", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--k_homomorphs", type=int, default=10)
    ap.add_argument("--include_self_matches", action="store_true")
    ap.add_argument("--out_dir", default="homomorphisms_csv")
    args = ap.parse_args()

    types1, emb1 = load_nat(args.cache_dir, args.nat1)
    types2, emb2 = load_nat(args.cache_dir, args.nat2)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    extractor = HomomorphismExtractor(
        k_homomorphs=args.k_homomorphs,
        include_self_matches=args.include_self_matches
    )

    sim = extractor.compute_similarity_matrix(emb1, emb2)
    rows = extractor.extract_homomorphisms_to_rows(sim, types1, types2, args.nat1, args.nat2)

    out = Path(args.out_dir) / f"{args.nat1}_{args.nat2}.csv"
    extractor.save_homomorphisms_to_csv(rows, str(out))
    logging.info(f"Done {args.nat1}-{args.nat2} -> {out}")

if __name__ == "__main__":
    main()
