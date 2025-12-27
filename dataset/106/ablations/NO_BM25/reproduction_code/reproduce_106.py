import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

batch_size = 8
input_dim = (3, 64, 64)
codebook_size = 256
codebook_dim = 64

input_data = torch.randn(batch_size, *input_dim)
vq = VectorQuantize(dim=3, codebook_size=codebook_size, codebook_dim=codebook_dim)
codebook = torch.randn(codebook_size, codebook_dim)
vq.codebook = codebook

quantized_output, indices, distances = vq(input_data)

if torch.isnan(distances).any():
    print("NaN values found in distances tensor")