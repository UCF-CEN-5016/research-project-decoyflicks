import torch

class DummyEncoder:
    def __init__(self, seq_len, embed_dim):
        # pos_embedding shape: (1, seq_len, embed_dim)
        self.pos_embedding = torch.randn(1, seq_len, embed_dim)

class DummyModel:
    def __init__(self, seq_len, embed_dim):
        self.encoder = DummyEncoder(seq_len, embed_dim)

def reproduce_bug():
    embed_dim = 768
    seq_len = 197  # 196 patches + 1 cls token
    model = DummyModel(seq_len, embed_dim)

    num_patches = 196  # typical for 14x14 patches

    # tokens shape: (1, num_patches+1, embed_dim)
    tokens = torch.randn(1, num_patches + 1, embed_dim)

    # Correct indexing should be:
    # tokens = tokens + model.encoder.pos_embedding[:, 1:(num_patches + 1), :]

    # Intentionally slicing with out-of-range index to trigger potential error
    # For example, num_patches+1 > seq_len
    try:
        out_of_range_num_patches = 200
        tokens_bad = tokens.clone()
        tokens_bad = tokens_bad + model.encoder.pos_embedding[:, 1:(out_of_range_num_patches + 1), :]
    except Exception as e:
        print("Error triggered:", e)

if __name__ == "__main__":
    reproduce_bug()