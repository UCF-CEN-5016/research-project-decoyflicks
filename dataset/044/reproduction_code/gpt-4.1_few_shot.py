import os
import torch

# Simulate the minimal inference that leads to empty text after OOV filtering
# Assume model and tokenizer are loaded from downloaded jvn.tar.gz (mocked here)

class MockTokenizer:
    def __init__(self):
        # Example vocab without Japanese tokens, causing all tokens to be OOV
        self.vocab = {"<pad>": 0, "<unk>": 1}

    def text_to_tokens(self, text):
        # All tokens will be OOV since vocab is minimal
        tokens = []
        for ch in text:
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            else:
                tokens.append(self.vocab["<unk>"])
        return tokens

    def filter_oov(self, tokens):
        # Filter out all OOV tokens (all are <unk>)
        filtered = [t for t in tokens if t != self.vocab["<unk>"]]
        return filtered

# Simulate input Japanese text that is entirely OOV
input_text = "こんにちは"  # Japanese greeting

tokenizer = MockTokenizer()
tokens = tokenizer.text_to_tokens(input_text)
filtered_tokens = tokenizer.filter_oov(tokens)

print("Original tokens:", tokens)
print("Tokens after filtering OOV:", filtered_tokens)

# Output the text after filtering OOV (empty expected)
if not filtered_tokens:
    print("text after filtering OOV:")
else:
    # Normally convert tokens back to text (not needed here)
    print("text after filtering OOV:", filtered_tokens)