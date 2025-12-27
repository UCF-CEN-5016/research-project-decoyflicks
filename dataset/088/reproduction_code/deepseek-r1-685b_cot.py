import torch
import torch.nn.functional as F

def align_right(x, lens, pad_id=1):
    # The bug is here - pad_id is ignored and hardcoded 0 is used
    padded = F.pad(x, (0, x.shape[-1] - lens), value=0)
    return padded

# Test case
seqs = torch.tensor([
    [1, 2, 3, 0, 0],  # seq1 (len=3)
    [4, 5, 0, 0, 0],  # seq2 (len=2)
    [6, 7, 8, 9, 0]   # seq3 (len=4)
])
lengths = torch.tensor([3, 2, 4])

# Try to pad with pad_id=1 (should use this value)
result = align_right(seqs, lengths, pad_id=1)

print("Original sequences:")
print(seqs)
print("\nExpected right-aligned with pad_id=1:")
print(torch.tensor([
    [0, 0, 1, 2, 3],
    [0, 0, 0, 4, 5],
    [0, 6, 7, 8, 9]
]))
print("\nActual result (bug - uses 0 instead of 1):")
print(result)