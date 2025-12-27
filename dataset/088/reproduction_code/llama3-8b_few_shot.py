import torch
from x_transformers import AutoRegressiveWrapper

# Define a simple transformer model
model = torch.nn.Transformer()

# Use the wrapper to align right
wrapper = AutoRegressiveWrapper(model, pad_id=0)  # Set pad ID

# Create some input sequences
sequences = [
    ["hello", "world"],
    ["foo", "bar"],
    ["baz"]
]

# Process the sequences using the wrapper
results = []
for sequence in sequences:
    padded_sequence = wrapper.align_right(sequence)
    results.append(padded_sequence)

print("Results:", results)