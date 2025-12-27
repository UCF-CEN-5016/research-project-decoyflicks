import torch
from torch.nn import Embedding

class TTSModel:
    def __init__(self, vocab_size, embedding_dim):
        self.model = Embedding(vocab_size, embedding_dim)

    def filter_oov(self, text):
        def char_to_idx(c):
            return ord(c) - ord('A')

        filtered = [c for c in text if char_to_idx(c) < self.model.num_embeddings]
        return ''.join(filtered)

# Mock TTS model with limited vocabulary
vocab_size = 100
embedding_dim = 256
tts_model = TTSModel(vocab_size, embedding_dim)

# Sample input text with OOV characters
input_text = "XYZabc123"

# Process input text
filtered_text = tts_model.filter_oov(input_text)
print(f"Original text: {input_text}")
print(f"Filtered text: {filtered_text}")  # Should be empty if all chars are OOV