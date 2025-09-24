import re
import string

import datasets
from nltk.tokenize import word_tokenize


def load_responses(path):
    data = datasets.load_from_disk(path)
    # print(f"Loaded dataset: {data}")
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
