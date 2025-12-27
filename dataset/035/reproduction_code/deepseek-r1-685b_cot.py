import torch
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Sample input with 4 features (dim=4)
x = torch.randn(1, 5, 4)  # (batch_size, seq_len, d_model)

# Incorrect initialization (dim=3) - will cause error
try:
    rotary_pe_wrong = RotaryPositionalEmbeddings(3)
    out_wrong = rotary_pe_wrong(x)
    print("Worked with dim=3 (shouldn't happen)")
except Exception as e:
    print(f"Error with dim=3 (expected): {str(e)}")

# Correct initialization (dim=4)
rotary_pe_correct = RotaryPositionalEmbeddings(4)
out_correct = rotary_pe_correct(x)
print("Successfully processed with dim=4")
print("Output shape:", out_correct.shape)