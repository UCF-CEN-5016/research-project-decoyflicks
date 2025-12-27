import torch
from torch.nn import Embedding
from typing import Iterable

class TTSModel:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.embedding = Embedding(vocab_size, embedding_dim)

    @staticmethod
    def _char_to_index(char: str) -> int:
        return ord(char) - ord('A')

    def filter_oov_chars(self, text: str) -> str:
        max_embeddings = self.embedding.num_embeddings
        filtered_chars: Iterable[str] = (
            ch for ch in text if self._char_to_index(ch) < max_embeddings
        )
        return ''.join(filtered_chars)

if __name__ == "__main__":
    # Mock TTS model with limited vocabulary
    vocab_size = 100
    embed_dim = 256
    tts_model = TTSModel(vocab_size, embed_dim)

    # Sample input text with OOV characters
    input_text = "XYZabc123"

    # Process input text
    filtered_text = tts_model.filter_oov_chars(input_text)
    print(f"Original text: {input_text}")
    print(f"Filtered text: {filtered_text}")  # Should be empty if all chars are OOV