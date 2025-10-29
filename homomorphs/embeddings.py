import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import defaultdict
import logging
import sys
import json
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nationality_country import nationality_to_country
except ImportError:
    logging.warning("Could not import nationality_to_country mapping.")
    nationality_to_country = {}

logging.basicConfig(level=logging.INFO)


class EmbeddingGenerator:
    def __init__(self, model_name="bert-base-uncased", device=None, tokenization_path=None):
        if tokenization_path is None:
            raise ValueError("tokenization_path is required. Run build_npmi_tokenization.py first.")
        
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
        self.tokenization_data = self._load_tokenization_data(tokenization_path)
        self.lookup_structures = self._build_lookup_structures()
        
        logging.info(f"Loaded model {model_name} on device {self.device}")
        total_words = sum(len(words) for words in self.tokenization_data.values())
        logging.info(f"Loaded pre-computed tokenizations for {len(self.tokenization_data)} nationalities ({total_words} words)")

    # -------- cache helpers --------
    def _cache_dir_path(self, cache_dir: str) -> Path:
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _cache_path(self, nationality: str, cache_dir: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in nationality)
        return self._cache_dir_path(cache_dir) / f"{safe}.npz"

    def _try_load_cache(self, nationality: str, cache_dir: str) -> Tuple[List[str], np.ndarray]:
        f = self._cache_path(nationality, cache_dir)
        if not f.exists():
            return [], np.array([])
        try:
            data = np.load(f, allow_pickle=False)
            words = data["words"].astype(object).tolist()
            embs = data["embs"].astype(np.float32)  # upcast for cosine stability
            logging.info(f"Cache hit for {nationality}: {f} ({len(words)} words)")
            return words, embs
        except Exception as e:
            logging.warning(f"Failed to load cache for {nationality} at {f}: {e}")
            return [], np.array([])

    def _save_cache(self, nationality: str, words: List[str], embs: np.ndarray, cache_dir: str):
        f = self._cache_path(nationality, cache_dir)
        np.savez_compressed(
            f,
            words=np.array(words, dtype=object),
            embs=embs.astype(np.float16) 
        )
        logging.info(f"Cache saved for {nationality}: {f} ({len(words)} words, dim={embs.shape[1] if embs.size else 0})")

    def _load_tokenization_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_lookup_structures(self):
        first_token_lookup = defaultdict(list)
        exact_token_lookup = defaultdict(list)
        
        for nationality, words in self.tokenization_data.items():
            for word, info in words.items():
                tokens = info['tokens']
                if len(tokens) == 1:
                    exact_token_lookup[tokens[0]].append((word, nationality))
                else:
                    first_token_lookup[tokens[0]].append((word, tokens, nationality))
        
        return {
            'first_token_lookup': dict(first_token_lookup),
            'exact_token_lookup': dict(exact_token_lookup)
        }

    def _match_token_sequence(self, tokens, start_pos, expected_tokens):
        if start_pos + len(expected_tokens) > len(tokens):
            return False
        return tokens[start_pos:start_pos + len(expected_tokens)] == expected_tokens

    def _fast_npmi_word_matching(self, tokens, embeddings, nationality, debug_words=None):
        if nationality not in self.tokenization_data:
            logging.warning(f"No tokenization data for nationality: {nationality}")
            return {}

        word_embeddings_map = defaultdict(list)

        nationality_single_tokens = set()
        nationality_first_tokens = set()

        for word, info in self.tokenization_data[nationality].items():
            tokens_for_word = info['tokens']
            if len(tokens_for_word) == 1:
                nationality_single_tokens.add(tokens_for_word[0])
            else:
                nationality_first_tokens.add(tokens_for_word[0])

        logging.info(f"Nationality-specific tokens: {len(nationality_single_tokens)} single, {len(nationality_first_tokens)} first-tokens")

        # Step 1: Single-token
        single_token_count = 0
        for i, token in enumerate(tokens):
            if token in nationality_single_tokens:
                for word, word_nationality in self.lookup_structures['exact_token_lookup'][token]:
                    if word_nationality == nationality:
                        word_embeddings_map[word].append(embeddings[i])
                        single_token_count += 1
                        if debug_words and word in debug_words:
                            logging.info(f"DEBUG: Found single-token match for '{word}' at position {i}")

        logging.info(f"Found {single_token_count} single-token matches")

        # Step 2: Multi-token
        multi_token_count = 0
        i = 0
        while i < len(tokens):
            current_token = tokens[i]
            if current_token in nationality_first_tokens:
                for word, expected_tokens, word_nationality in self.lookup_structures['first_token_lookup'][current_token]:
                    if word_nationality != nationality:
                        continue
                    if self._match_token_sequence(tokens, i, expected_tokens):
                        word_embs = embeddings[i:i+len(expected_tokens)]
                        word_embeddings_map[word].append(np.mean(word_embs, axis=0))
                        multi_token_count += 1
                        if debug_words and word in debug_words:
                            logging.info(f"DEBUG: Matched '{word}' at position {i} using tokens: {tokens[i:i+len(expected_tokens)]}")
                        i += len(expected_tokens)
                        break
                else:
                    i += 1
            else:
                i += 1

        if debug_words:
            for word in debug_words:
                if word not in word_embeddings_map:
                    logging.warning(f"DEBUG: '{word}' was NOT matched in token stream despite being in tokenization and appearing in story text.")

        logging.info(f"Found {multi_token_count} multi-token matches")
        total_found = sum(len(v) for v in word_embeddings_map.values())
        logging.info(f"Total word instances found: {total_found}")

        return word_embeddings_map

    def get_embeddings_batched(self, sentences, batch_size=32):
        all_words = []
        all_embeddings = []

        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
            batch_texts = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden_states = outputs.last_hidden_state.cpu()

            for tkseq, embseq in zip(inputs.input_ids, last_hidden_states):
                for tk, emb in zip(self.tokenizer.convert_ids_to_tokens(tkseq), embseq):
                    if tk not in ['[PAD]', '[CLS]', '[SEP]']:
                        all_words.append(tk)
                        all_embeddings.append(emb.numpy())

        return all_words, np.array(all_embeddings)
    
    def prepare_story_data_for(self, dataset, nationality, n_columns=5):
        """
        Return a story_dict containing only the given nationality.
        """
        from collections import defaultdict
        story_dict = defaultdict(list)
        for ex in dataset:
            if ex.get("nationality") != nationality:
                continue
            for i in range(n_columns):
                col = f"tokenized_response_{i}"
                if col in ex and ex[col]:
                    if isinstance(ex[col], list):
                        text = " ".join(ex[col]).lower()
                    else:
                        text = str(ex[col]).lower()
                    story_dict[nationality].append(text)
        return dict(story_dict)


    def prepare_story_data(self, dataset, n_columns=5):
        story_dict = defaultdict(list)
        for example in tqdm(dataset, desc="Preparing story data"):
            nationality = example["nationality"]
            for i in range(n_columns):
                column_name = f"tokenized_response_{i}"
                if column_name in example and example[column_name]:
                    if isinstance(example[column_name], list):
                        text = " ".join(example[column_name]).lower()
                        story_dict[nationality].append(text)
                    else:
                        story_dict[nationality].append(example[column_name].lower())
        return dict(story_dict)

    def _extract_relevant_country_patterns(self, nationality):
        if not nationality_to_country or nationality not in nationality_to_country:
            return {}

        country_variants = nationality_to_country[nationality].split(',')
        canonical_name = country_variants[0].strip().lower().replace(' ', '_')
        patterns = [variant.strip().lower().split() for variant in country_variants if len(variant.strip().split()) > 1]
        patterns.sort(key=len, reverse=True)
        return {canonical_name: patterns}

    def _detect_and_merge_nationality_specific_countries(self, words, embeddings, nationality):
        patterns = self._extract_relevant_country_patterns(nationality)
        if not patterns:
            return words, embeddings

        merged_words = []
        merged_embeddings = []
        i = 0

        while i < len(words):
            matched = False
            for canonical_name, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if i + len(pattern) <= len(words):
                        sequence = [words[i + j].lower().lstrip('#') for j in range(len(pattern))]
                        if sequence == pattern:
                            merged_words.append(canonical_name)
                            merged_embeddings.append(np.mean(embeddings[i:i + len(pattern)], axis=0))
                            i += len(pattern)
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                merged_words.append(words[i])
                merged_embeddings.append(embeddings[i])
                i += 1

        return merged_words, np.array(merged_embeddings)

    def generate_nationality_embeddings(self, story_dict, nationality, batch_size=32, debug_words=None, cache_dir="cache_npmis"):
        cached_words, cached_embs = self._try_load_cache(nationality, cache_dir)
        if len(cached_words) > 0 and cached_embs.size > 0:
            return cached_words, cached_embs

        import time
        start_time = time.time()

        if nationality not in self.tokenization_data:
            logging.warning(f"No tokenization data for nationality: {nationality}")
            return [], np.array([])

        target_words = set(self.tokenization_data[nationality].keys())
        logging.info(f"Target NPMI words for {nationality}: {len(target_words)} words")

        if debug_words:
            logging.info(f"=== DEBUG: checking {debug_words} in {nationality} ===")
            for word in debug_words:
                if word in self.tokenization_data[nationality]:
                    tokens = self.tokenization_data[nationality][word]['tokens']
                    logging.info(f"DEBUG: '{word}' tokenized as {tokens}")
                else:
                    logging.info(f"DEBUG: '{word}' not found in tokenization data")

            logging.info("=== DEBUG: scanning raw stories for debug words ===")
            for story in story_dict.get(nationality, []):
                for word in debug_words:
                    if word in story:
                        logging.info(f"DEBUG: Found raw string '{word}' in story: {story[:100]}...")

        # Convert stories to sentences
        sentences = []
        for story in story_dict.get(nationality, []):
            sentences.extend(sent_tokenize(story))
        logging.info(f"{nationality}: {len(sentences)} sentences")

        words, embeddings = self.get_embeddings_batched(sentences, batch_size)
        words, embeddings = self._detect_and_merge_nationality_specific_countries(words, embeddings, nationality)
        word_embeddings_map = self._fast_npmi_word_matching(words, embeddings, nationality, debug_words)

        word_types = []
        word_embeddings_list = []
        for word in target_words:
            if word in word_embeddings_map:
                avg_embedding = np.mean(word_embeddings_map[word], axis=0)
                word_types.append(word)
                word_embeddings_list.append(avg_embedding)

        logging.info(f"Generated embeddings for {len(word_types)} out of {len(target_words)} NPMI words")
        logging.info(f"Total time: {time.time() - start_time:.2f} seconds")

        embs = np.array(word_embeddings_list, dtype=np.float32)
        self._save_cache(nationality, word_types, embs, cache_dir)
        return word_types, embs
