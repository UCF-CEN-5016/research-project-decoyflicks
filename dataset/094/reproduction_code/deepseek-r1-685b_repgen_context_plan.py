import torch
from x_transformers import ContinuousTransformerWrapper, Decoder

def create_memory(num_layers, batch_size, mem_length, hidden_dim):
    return [torch.zeros(batch_size, mem_length, hidden_dim) for _ in range(num_layers)]

# Initialize model with memory settings
model = ContinuousTransformerWrapper(
    dim_out=256,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=8,
        rotary_pos_emb=True
    )
)

# Create test inputs
batch_size = 1
seq_length = 1024
hidden_dim = 512

x = torch.randn(batch_size, seq_length, hidden_dim)  # (batch, seq, dim)
mask = torch.rand(batch_size, seq_length) > 0.5  # Random mask
num_mem_layers = 6
mem_length = 100
mems = create_memory(num_mem_layers, batch_size, mem_length, hidden_dim)  # Initial memory

# Test the model with different memory settings
def test_model(model, x, mask, mems):
    try:
        logits, new_mems = model(x, mask=mask, mems=mems, return_mems=True)
        print("Success!")
        print("Logits shape:", logits.shape)
        print("Memory shapes:", [m.shape for m in new_mems])
    except Exception as e:
        print("Failed with error:", e)

# Test with initial memory provided
test_model(model, x, mask, mems)

# Test with mems=None
logits, new_mems = model(x, mask=mask, mems=None, return_mems=True)
print("Works with mems=None")
print("Logits shape:", logits.shape)
print("Memory shapes:", [m.shape for m in new_mems])