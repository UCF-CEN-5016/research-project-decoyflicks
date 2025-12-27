import torch
from vector_quantize_pytorch import VectorQuantizer, LFQ, FSQ

batch_size = 1
height, width, channels = 32, 32, 3
input_data = torch.randn(batch_size, height, width, channels)

# Initialize the VectorQuantizer
vq = VectorQuantizer(codebook_size=65536, dim=16, entropy_loss_weight=0.1, diversity_gamma=1.)

# Forward pass through the VectorQuantizer
output_vq = vq(input_data)
print(output_vq.isnan().any())

# Monitor GPU memory usage (requires torch.cuda.memory_allocated and torch.cuda.max_memory_allocated)
vq_gpu_usage = torch.cuda.memory_allocated() - torch.cuda.max_memory_allocated(-1)
print(f"VectorQuantizer GPU Memory Usage: {vq_gpu_usage} bytes")

# Initialize the LFQ
lfq = LFQ(codebook_size=65536, dim=16, entropy_loss_weight=0.1, diversity_gamma=1.)

# Forward pass through the LFQ
output_lfq = lfq(input_data)
print(output_lfq.isnan().any())

# Monitor GPU memory usage (requires torch.cuda.memory_allocated and torch.cuda.max_memory_allocated)
lfq_gpu_usage = torch.cuda.memory_allocated() - torch.cuda.max_memory_allocated(-1)
print(f"LFQ GPU Memory Usage: {lfq_gpu_usage} bytes")

# Initialize the FSQ
fsq = FSQ(levels=[8,5,5,5], return_indices=True)

# Forward pass through the FSQ
output_fsq, indices = fsq(input_data)
print(output_fsq.isnan().any())

# Monitor GPU memory usage (requires torch.cuda.memory_allocated and torch.cuda.max_memory_allocated)
fsq_gpu_usage = torch.cuda.memory_allocated() - torch.cuda.max_memory_allocated(-1)
print(f"FSQ GPU Memory Usage: {fsq_gpu_usage} bytes")