#!/usr/bin/env python3
"""
Build tokenization lookup table for NPMI words.
Pre-tokenizes all NPMI words and saves the mappings for fast lookup during embedding generation.
"""

import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


def build_tokenization_lookup(npmi_path, model_name="bert-base-uncased", output_path=None):
    """
    Build tokenization lookup table for all NPMI words.
    
    Args:
        npmi_path (str): Path to NPMI JSON file
        model_name (str): HuggingFace model name for tokenizer
        output_path (str): Output path for tokenization lookup (default: npmi_path + '.tokenized')
        
    Returns:
        str: Path to saved tokenization file
    """
    # Load tokenizer
    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load NPMI data
    logging.info(f"Loading NPMI data from: {npmi_path}")
    with open(npmi_path, 'r', encoding='utf-8') as f:
        npmi_data = json.load(f)
    
    # Build tokenization lookup
    tokenization_lookup = {}
    total_words = sum(len(words) for words in npmi_data.values())
    
    logging.info(f"Tokenizing {total_words} words across {len(npmi_data)} nationalities...")
    
    with tqdm(total=total_words, desc="Tokenizing words") as pbar:
        for nationality, word_scores in npmi_data.items():
            tokenization_lookup[nationality] = {}
            
            for word, score in word_scores.items():
                # Tokenize the word
                tokens = tokenizer.tokenize(word.lower())
                
                # Store the tokenization
                tokenization_lookup[nationality][word.lower()] = {
                    'tokens': tokens,
                    'score': score,
                    'token_count': len(tokens)
                }
                
                pbar.update(1)
    
    # Save tokenization lookup
    if output_path is None:
        raise ValueError("output_path must be specified as a directory")
    
    # output_path is a folder, create filename based on input
    from pathlib import Path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(npmi_path)
    output_path = output_dir / f"{input_path.stem}_tokenized.json"
    
    logging.info(f"Saving tokenization lookup to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenization_lookup, f, indent=2, ensure_ascii=False)
    
    return output_path


def load_tokenization_lookup(tokenization_path):
    """
    Load pre-computed tokenization lookup.
    
    Args:
        tokenization_path (str): Path to tokenization JSON file
        
    Returns:
        dict: Tokenization lookup data
    """
    with open(tokenization_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_fast_lookup_structures(tokenization_data):
    """
    Create optimized lookup structures for fast matching during embedding generation.
    
    Args:
        tokenization_data (dict): Loaded tokenization data
        
    Returns:
        dict: Fast lookup structures
    """
    # Create reverse lookup: first_token -> [(word, full_tokens, nationality)]
    first_token_lookup = defaultdict(list)
    
    # Create exact token lookup: token -> [(word, nationality)]
    exact_token_lookup = defaultdict(list)
    
    for nationality, words in tokenization_data.items():
        for word, info in words.items():
            tokens = info['tokens']
            
            if len(tokens) == 1:
                # Single token - add to exact lookup
                exact_token_lookup[tokens[0]].append((word, nationality))
            else:
                # Multi-token - add to first token lookup
                first_token_lookup[tokens[0]].append((word, tokens, nationality))
    
    return {
        'first_token_lookup': dict(first_token_lookup),
        'exact_token_lookup': dict(exact_token_lookup),
        'tokenization_data': tokenization_data
    }


def main():
    parser = argparse.ArgumentParser(description="Build tokenization lookup for NPMI words")
    parser.add_argument("--npmi_path", required=True, help="Path to NPMI JSON file")
    parser.add_argument("--model_name", default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--output_path", help="Output path for tokenization lookup")
    
    args = parser.parse_args()
    
    # Build tokenization lookup
    output_path = build_tokenization_lookup(
        npmi_path=args.npmi_path,
        model_name=args.model_name,
        output_path=args.output_path
    )
    
    logging.info(f"Tokenization lookup saved to: {output_path}")
    print(f"Use this file with: --tokenization_path {output_path}")


if __name__ == "__main__":
    main()