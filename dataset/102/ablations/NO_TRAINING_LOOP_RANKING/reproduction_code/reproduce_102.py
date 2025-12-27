import torch
from vector_quantize_pytorch import ResidualSimVQ

def main():
    batch_size = 1
    input_dim = (512, 32, 32)
    
    residual_sim_vq = ResidualSimVQ(
        dim=512,
        num_quantizers=4,
        codebook_size=1024,
        channel_first=True
    )
    
    x = torch.randn(batch_size, *input_dim)
    
    try:
        quantized, indices, commit_loss = residual_sim_vq(x)
    except NameError as e:
        print(e)

if __name__ == "__main__":
    main()