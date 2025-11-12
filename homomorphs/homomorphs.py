import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging, json

logging.basicConfig(level=logging.INFO)


class HomomorphismExtractor:
    def __init__(self, k_homomorphs=10, include_self_matches=False):
        self.k_homomorphs = int(k_homomorphs)
        self.include_self_matches = bool(include_self_matches)

    @staticmethod
    def _validate_embs(emb1: np.ndarray, emb2: np.ndarray):
        if emb1 is None or emb2 is None:
            raise ValueError("embeddings cannot be None")
        if emb1.size == 0 or emb2.size == 0:
            raise ValueError("empty embeddings")
        if emb1.ndim != 2 or emb2.ndim != 2:
            raise ValueError("embeddings must be 2D arrays")
        if emb1.shape[1] != emb2.shape[1]:
            raise ValueError(f"dim mismatch: {emb1.shape} vs {emb2.shape}")

    def compute_similarity_matrix(self, emb1, emb2):
        """
        Compute cosine similarity matrix between two sets of embeddings.
        emb1: (n1, d), emb2: (n2, d) -> (n1, n2)
        """
        self._validate_embs(emb1, emb2)
        logging.info(f"Computing similarity matrix: {emb1.shape[0]} x {emb2.shape[0]}")
        return cosine_similarity(emb1, emb2)

    @staticmethod
    def _topk_indices(sim_vec: np.ndarray, k: int):
        """Return indices of top-k values in descending order."""
        k = min(k, sim_vec.shape[0])
        if k <= 0:
            return np.array([], dtype=int)
        # argpartition for O(n), then sort the slice descending
        part = np.argpartition(sim_vec, -k)[-k:]
        return part[np.argsort(sim_vec[part])[::-1]]

    def extract_from_similarity_matrix(self, similarity_matrix, word_types1, word_types2, nat1, nat2):
        """
        Extract homomorphisms from a precomputed similarity matrix.
        """
        sim = np.asarray(similarity_matrix)
        n1, n2 = sim.shape
        assert n1 == len(word_types1) and n2 == len(word_types2), "shape/label mismatch"

        rows = []

        # nat1 -> nat2
        for i, w1 in enumerate(word_types1):
            vec = sim[i].copy()
            if not self.include_self_matches:
                # mask exact same surface form if present on target side
                mask = np.array([w2 != w1 for w2 in word_types2], dtype=bool)
                if not mask.any():
                    continue
                vec[~mask] = -np.inf
            idx = self._topk_indices(vec, self.k_homomorphs)
            rows.append({
                "word": w1,
                "source_nationality": nat1,
                "target_nationality": nat2,
                "homomorphs": [word_types2[j] for j in idx],
                "similarities": [float(vec[j]) for j in idx],
            })

        # nat2 -> nat1
        for j, w2 in enumerate(word_types2):
            vec = sim[:, j].copy()
            if not self.include_self_matches:
                mask = np.array([w1 != w2 for w1 in word_types1], dtype=bool)
                if not mask.any():
                    continue
                vec[~mask] = -np.inf
            idx = self._topk_indices(vec, self.k_homomorphs)
            rows.append({
                "word": w2,
                "source_nationality": nat2,
                "target_nationality": nat1,
                "homomorphs": [word_types1[i] for i in idx],
                "similarities": [float(vec[i]) for i in idx],
            })

        return rows

    def extract_homomorphisms_to_rows(self, similarity_matrix, types1, types2, nat1, nat2):
        """
        Backward-compatible extractor using the same top-k strategy and masking.
        """
        return self.extract_from_similarity_matrix(similarity_matrix, types1, types2, nat1, nat2)

    def process_nationality_pair(self, embedding_gen, story_dict, nat1, nat2, batch_size=32):
        """
        Legacy path that recomputes embeddings. .
        """
        logging.info(f"Processing pair: {nat1} vs {nat2}")
        types1, emb1 = embedding_gen.generate_nationality_embeddings(story_dict, nat1, batch_size)
        types2, emb2 = embedding_gen.generate_nationality_embeddings(story_dict, nat2, batch_size)
        sim = self.compute_similarity_matrix(emb1, emb2)
        rows = self.extract_homomorphisms_to_rows(sim, types1, types2, nat1, nat2)
        logging.info(f"Extracted {len(rows)} rows for {nat1}-{nat2}")
        return rows

    def save_homomorphisms_to_csv(self, homomorphism_rows, output_file):
        """
        Save lists via JSON strings to avoid unsafe eval.
        """
        recs = []
        for r in homomorphism_rows:
            recs.append({
                "word": r["word"],
                "source_nationality": r["source_nationality"],
                "target_nationality": r["target_nationality"],
                "homomorphs": json.dumps(r["homomorphs"], ensure_ascii=False),
                "similarities": json.dumps([round(float(s), 6) for s in r["similarities"]]),
            })
        df = pd.DataFrame(recs)
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(recs)} homomorphisms to {output_file}")

    def load_homomorphisms_from_csv(self, input_file):
        """
        Load JSON-encoded list fields back.
        """
        df = pd.read_csv(input_file)
        out = []
        for _, row in df.iterrows():
            out.append({
                "word": row["word"],
                "source_nationality": row["source_nationality"],
                "target_nationality": row["target_nationality"],
                "homomorphs": json.loads(row["homomorphs"]),
                "similarities": [float(x) for x in json.loads(row["similarities"])],
            })
        return out
