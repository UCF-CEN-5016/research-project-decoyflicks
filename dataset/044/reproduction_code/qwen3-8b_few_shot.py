import torch
from torch.nn import Embedding

# Mock TTS model with limited vocabulary
vocab_size = 100  # Model's vocabulary size
embedding_dim = 256
model = Embedding(vocab_size, embedding_dim)

# Sample input text with OOV characters (characters > vocab_size-1)
input_text = "XYZabc123"  # Assume 'X', 'Y', 'Z' are OOV

# Simulate OOV filtering process
def filter_oov(text, vocab_size):
    # Convert text to indices (mock tokenizer)
    def char_to_idx(c):
        return ord(c) - ord('A')  # Simple mapping for demo
    
    # Filter out characters with indices >= vocab_size
    filtered = []
    for c in text:
        idx = char_to_idx(c)
        if idx < vocab_size:
            filtered.append(c)
    return ''.join(filtered)

# Process input text
filtered_text = filter_oov(input_text, vocab_size)
print(f"Original text: {input_text}")
print(f"Filtered text: {filtered_text}")  # Should be empty if all chars are OOV