import torch
from vector_quantize_pytorch import ResidualLFQ, get_codebook


def build_input(seq_len: int, dim: int, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(seq_len, dim)


def prepare_codebook(num_embeddings: int, dim: int):
    codebook, _ = get_codebook(num_embeddings=num_embeddings, dim=dim)
    return codebook


def build_model(dim: int, num_heads: int, num_layers: int, codebook, use_mask: bool, commitment_loss_weight: float):
    return ResidualLFQ(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        codebook=codebook,
        use_mask=use_mask,
        commitment_loss_weight=commitment_loss_weight,
    )


def run_demo():
    dim = 14
    seq_len = 3700
    num_embeddings = 256
    num_heads = 8
    num_layers = 2
    commitment_loss_weight = 1.0
    use_mask = True

    x = build_input(seq_len=seq_len, dim=dim)
    codebook = prepare_codebook(num_embeddings=num_embeddings, dim=dim)
    model = build_model(dim=dim, num_heads=num_heads, num_layers=num_layers, codebook=codebook,
                        use_mask=use_mask, commitment_loss_weight=commitment_loss_weight)

    output = model(x)
    print(output.shape)


if __name__ == "__main__":
    run_demo()