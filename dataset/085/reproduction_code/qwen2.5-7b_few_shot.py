import torch

def add_pos_embedding(tokens, pos_embedding):
    # Add positional embeddings to tokens
    tokens = tokens + pos_embedding[:, :tokens.size(1)]
    return tokens

if __name__ == "__main__":
    # Create tokens and positional embeddings
    tokens = torch.randn(2, 3, 64)  # batch_size=2, num_patches=3, d=64
    pos_embedding = torch.randn(1, 4, 64)  # num_patches + 1 = 4

    # Add positional embeddings to tokens
    tokens = add_pos_embedding(tokens, pos_embedding)

    print(tokens.shape)