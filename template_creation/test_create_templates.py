import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "/home/shailyjb/converging_cultural_narratives")

from nationality_country import nationality_to_country
from template_creation.create_templates import (
    compress_masked_response,
    get_masking_words,
    mask_country_nationality,
    mask_response,
)


class TestCreateTemplates(unittest.TestCase):

    def test_mask_country_nationality_united_states(self):
        """Test masking for American nationality with various country variants"""
        nationality = "American"

        # Test case 1: "United States of America"
        input_tokens = [
            "i",
            "am",
            "from",
            "the",
            "united",
            "states",
            "of",
            "america",
            "and",
            "i",
            "love",
            "it",
        ]
        expected = ["i", "am", "from", "the", "[MASK]", "and", "i", "love", "it"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

        # Test case 2: "United States"
        input_tokens = [
            "i",
            "live",
            "in",
            "united",
            "states",
            "and",
            "it",
            "is",
            "great",
        ]
        expected = ["i", "live", "in", "[MASK]", "and", "it", "is", "great"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

        # Test case 3: "USA"
        input_tokens = ["the", "usa", "is", "great"]
        expected = ["the", "[MASK]", "is", "great"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

        # Test case 4: "US"
        input_tokens = ["i", "am", "american", "and", "from", "the", "us"]
        expected = ["i", "am", "[MASK]", "and", "from", "[MASK]"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

        # Test case 5: Multiple occurrences
        input_tokens = ["united", "states", "and", "usa", "are", "the", "same"]
        expected = ["[MASK]", "and", "[MASK]", "are", "the", "same"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

    def test_mask_country_nationality_multi_word_countries(self):
        """Test masking for countries with multiple words"""

        # Test Saint Kitts and Nevis
        nationality = "Kittitian"
        input_tokens = ["i", "am", "from", "saint", "kitts", "and", "nevis"]
        expected = ["i", "am", "from", "[MASK]"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

        # Test Bosnia and Herzegovina
        nationality = "Bosnian"
        input_tokens = ["bosnia", "and", "herzegovina", "is", "beautiful"]
        expected = ["[MASK]", "is", "beautiful"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

    def test_mask_country_nationality_case_insensitive(self):
        """Test that masking is case insensitive"""
        nationality = "American"

        # Mixed case
        input_tokens = ["I", "am", "from", "UNITED", "states", "Of", "America"]
        expected = ["I", "am", "from", "[MASK]"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

    def test_mask_country_nationality_no_matches(self):
        """Test when no country/nationality matches are found"""
        nationality = "American"
        input_tokens = ["i", "like", "pizza", "and", "pasta"]
        expected = ["i", "like", "pizza", "and", "pasta"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

    def test_mask_country_nationality_existing_masks(self):
        """Test that existing mask tokens are preserved"""
        nationality = "American"
        input_tokens = ["i", "am", "[MASK]", "from", "united", "states"]
        expected = ["i", "am", "[MASK]", "from", "[MASK]"]
        result = mask_country_nationality(input_tokens, nationality)
        self.assertEqual(result, expected)

    def test_mask_country_nationality_custom_mask_token(self):
        """Test with custom mask token"""
        nationality = "American"
        input_tokens = ["i", "am", "from", "usa"]
        expected = ["i", "am", "from", "<MASKED>"]
        result = mask_country_nationality(
            input_tokens, nationality, mask_token="<MASKED>"
        )
        self.assertEqual(result, expected)

    def test_mask_response_basic(self):
        """Test basic word masking functionality"""
        response = ["I", "love", "pizza", "and", "burgers"]
        masking_words = ["pizza", "burgers"]
        expected = ["i", "love", "[MASK]", "and", "[MASK]"]
        result = mask_response(response, masking_words)
        self.assertEqual(result, expected)

    def test_mask_response_punctuation(self):
        """Test that punctuation is preserved"""
        response = ["Hello", ",", "world", "!"]
        masking_words = ["world"]
        expected = ["hello", ",", "[MASK]", "!"]
        result = mask_response(response, masking_words)
        self.assertEqual(result, expected)

    def test_mask_response_case_insensitive(self):
        """Test that masking is case insensitive"""
        response = ["I", "LOVE", "Pizza"]
        masking_words = ["love", "pizza"]
        expected = ["i", "[MASK]", "[MASK]"]
        result = mask_response(response, masking_words)
        self.assertEqual(result, expected)

    def test_mask_response_custom_mask_token(self):
        """Test mask_response with custom mask token"""
        response = ["I", "love", "pizza"]
        masking_words = ["pizza"]
        expected = ["i", "love", "<HIDDEN>"]
        result = mask_response(response, masking_words, mask_token="<HIDDEN>")
        self.assertEqual(result, expected)

    def test_compress_masked_response_basic(self):
        """Test basic compression of consecutive masks"""
        response = ["i", "[MASK]", "[MASK]", "love", "[MASK]", "food"]
        expected = ["i", "[MASK]", "love", "[MASK]", "food"]
        result = compress_masked_response(response)
        self.assertEqual(result, expected)

    def test_compress_masked_response_no_consecutive(self):
        """Test compression when no consecutive masks exist"""
        response = ["i", "[MASK]", "love", "[MASK]", "food"]
        expected = ["i", "[MASK]", "love", "[MASK]", "food"]
        result = compress_masked_response(response)
        self.assertEqual(result, expected)

    def test_compress_masked_response_all_masks(self):
        """Test compression of all masks"""
        response = ["[MASK]", "[MASK]", "[MASK]", "[MASK]"]
        expected = ["[MASK]"]
        result = compress_masked_response(response)
        self.assertEqual(result, expected)

    def test_compress_masked_response_empty(self):
        """Test compression of empty response"""
        response = []
        expected = []
        result = compress_masked_response(response)
        self.assertEqual(result, expected)

    def test_get_masking_words_basic(self):
        """Test basic masking words extraction"""
        npmi = {
            "American": {"pizza": 0.8, "burger": 0.7, "freedom": 0.6},
            "Italian": {"pasta": 0.9, "pizza": 0.8, "wine": 0.7},
        }
        percentage_threshold = 0.5  # 50%

        result = get_masking_words(npmi, percentage_threshold)

        # Should get top 50% of words for each nationality
        expected = {
            "American": ["pizza"],  # 1 out of 3 words (33% rounded up)
            "Italian": ["pasta"],  # 1 out of 3 words (33% rounded up)
        }
        self.assertEqual(result, expected)

    def test_get_masking_words_zero_threshold(self):
        """Test with zero threshold (no words masked)"""
        npmi = {
            "American": {"pizza": 0.8, "burger": 0.7},
            "Italian": {"pasta": 0.9, "wine": 0.8},
        }
        percentage_threshold = 0.0

        result = get_masking_words(npmi, percentage_threshold)

        expected = {"American": [], "Italian": []}
        self.assertEqual(result, expected)

    def test_get_masking_words_full_threshold(self):
        """Test with 100% threshold (all words masked)"""
        npmi = {
            "American": {"pizza": 0.8, "burger": 0.7},
            "Italian": {"pasta": 0.9, "wine": 0.8},
        }
        percentage_threshold = 1.0

        result = get_masking_words(npmi, percentage_threshold)

        expected = {"American": ["pizza", "burger"], "Italian": ["pasta", "wine"]}
        self.assertEqual(result, expected)

    def test_integration_mask_response_and_country_nationality(self):
        """Test integration of word masking and country/nationality masking"""
        nationality = "American"
        response = ["i", "love", "pizza", "in", "the", "united", "states"]
        masking_words = ["pizza"]

        # First apply word masking
        masked_response = mask_response(response, masking_words)
        expected_after_word_mask = [
            "i",
            "love",
            "[MASK]",
            "in",
            "the",
            "united",
            "states",
        ]
        self.assertEqual(masked_response, expected_after_word_mask)

        # Then apply country/nationality masking
        final_result = mask_country_nationality(masked_response, nationality)
        expected_final = ["i", "love", "[MASK]", "in", "the", "[MASK]"]
        self.assertEqual(final_result, expected_final)

    def test_priority_longest_country_names_first(self):
        """Test that longer country names are matched before shorter ones"""
        nationality = "American"

        # This should match "united states of america" as a whole, not "united states" + "of" + "america"
        input_tokens = [
            "from",
            "united",
            "states",
            "of",
            "america",
            "to",
            "united",
            "states",
        ]
        result = mask_country_nationality(input_tokens, nationality)

        # Both "united states of america" and "united states" should be masked
        expected = ["from", "[MASK]", "to", "[MASK]"]
        self.assertEqual(result, expected)


def run_manual_tests():
    """Manual test function to run specific test cases and print results"""
    print("=== Manual Test Results ===")

    # Test United States variants
    print("\n1. Testing United States variants:")
    nationality = "American"
    test_cases = [
        ["i", "am", "from", "the", "united", "states", "of", "america"],
        ["i", "live", "in", "united", "states"],
        ["the", "usa", "is", "great"],
        ["i", "am", "american", "and", "from", "the", "us"],
    ]

    for i, test_input in enumerate(test_cases, 1):
        result = mask_country_nationality(test_input, nationality)
        print(f"  Test {i}: {test_input}")
        print(f"  Result: {result}")

    # Test multi-word countries
    print("\n2. Testing multi-word countries:")
    test_cases = [
        (["from", "saint", "kitts", "and", "nevis"], "Kittitian"),
        (["bosnia", "and", "herzegovina", "rocks"], "Bosnian"),
        (["trinidad", "and", "tobago", "is", "nice"], "Trinidadian"),
    ]

    for test_input, nationality in test_cases:
        result = mask_country_nationality(test_input, nationality)
        print(f"  {nationality}: {test_input} -> {result}")

    # Test compression
    print("\n3. Testing compression:")
    test_input = ["i", "[MASK]", "[MASK]", "love", "[MASK]", "[MASK]", "[MASK]", "food"]
    result = compress_masked_response(test_input)
    print(f"  Input:  {test_input}")
    print(f"  Output: {result}")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)

    # Run manual tests
    run_manual_tests()
