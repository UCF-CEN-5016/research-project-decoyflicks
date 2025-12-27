import torch
from vector_quantize_pytorch import ResidualLFQ

def main():
    input_dim = 14
    num_embeddings = 128
    embedding_dim = 64
    commitment_loss_weight = 1.0
    num_layers = 2

    x = torch.randn(2, 1851, 1, input_dim)
    mask = torch.randint(0, 2, (2, 1851, 1, 1), dtype=torch.bool)

    lfq = ResidualLFQ(
        input_dim=input_dim,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_loss_weight=commitment_loss_weight,
        num_layers=num_layers,
    )

    lfq(x, mask=mask)

if __name__ == "__main__":
    main()