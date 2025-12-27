import torch
from x_transformers import ContinuousTransformerWrapper, Decoder

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
seq_len = 1024
dim = 512
x = torch.randn(batch_size, seq_len, dim)
mask = torch.rand(batch_size, seq_len) > 0.5  # Random mask
mems = [torch.zeros(batch_size, 100, dim) for _ in range(6)]  # Initial memory

# Test model with mems
def test_model_with_mems(model, x, mask, mems):
    try:
        logits, new_mems = model(x, mask=mask, mems=mems, return_mems=True)
        print("Success!")
        print("Logits shape:", logits.shape)
        print("Memory shapes:", [m.shape for m in new_mems])
    except Exception as e:
        print("Failed with error:", e)

# Test model with mems=None
logits, new_mems = model(x, mask=mask, mems=None, return_mems=True)
print("Works with mems=None")
print("Logits shape:", logits.shape)
print("Memory shapes:", [m.shape for m in new_mems])