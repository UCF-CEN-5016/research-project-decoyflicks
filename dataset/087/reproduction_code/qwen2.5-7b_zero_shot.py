import torch

def generate_alibi_pos(seq_len):
    return torch.arange(seq_len).view(1, 1, seq_len, 1).repeat(1, 8, 1, seq_len)

def generate_dummy_tensor(seq_len):
    return torch.rand(8, seq_len, seq_len)

def perform_elementwise_addition(tensor1, tensor2):
    try:
        result = tensor1 + tensor2
    except RuntimeError as e:
        print(f"Error: {e}")
        result = None
    return result

seq_len = 5

# Step 1: Generate a 4D tensor (alibi_pos)
alibi_pos = generate_alibi_pos(seq_len)

# Step 2: Generate a 3D tensor (dummy_tensor)
dummy_tensor = generate_dummy_tensor(seq_len)

# Step 3: Attempt to perform element-wise addition
result = perform_elementwise_addition(dummy_tensor, alibi_pos)