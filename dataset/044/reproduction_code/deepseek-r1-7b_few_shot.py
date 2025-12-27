# -*- coding: utf-8 -*-
import sys

def main():
    # Sample output containing OOV tokens (simulated)
    sample_output = """
        私は、猫を育 Let's see a translated text example...
    """

    # Translation dictionary for OOV words
    oov_translation = {
        "猫": "kitten",
        "Let's": "Let's",  # Assuming this is in the model's vocabulary
        "example...": ""
    }

    # Function to replace OOV tokens using regex substitution
    def replace_oov(text, translation_dict):
        for token, replacement in translation_dict.items():
            text = re.sub(r'\b' + re.escape(token) + r'\b', replacement, text)
        return text

    import re
    translated_text = replace_oov(sample_output, oov_translation)

    # Print the result and check if it's empty
    print(f"Translated Text: {translated_text}")
    if not translated_text.strip():
        print("Error: No valid translation found. OOV handling may be incorrect.")

if __name__ == "__main__":
    main()