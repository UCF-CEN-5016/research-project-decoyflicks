import torch
from vector_quantize_pytorch import ResidualLFQ

def main():
    # Sample input and mask
    input_tensor = torch.randn(2, 3, 14, 14)  # (batch, channels, height, width)
    mask = torch.randint(0, 2, (2, 14, 14)).bool()  # (batch, height, width)

    # Initialize ResidualLFQ with commitment loss weight
    lfq = ResidualLFQ(
        dim=3,                  # Input channels
        codebook_size=10,       # Number of codebook vectors
        commitment_loss_weight=1.0  # Enable commitment loss
    )

    # Forward pass with mask
    output = lfq(input_tensor, mask=mask)

    # The following line is illustrative; actual computation may trigger the warning
    # The mismatch occurs when reshaping original_input vs quantized tensor
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()