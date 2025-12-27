import torch
import torch.nn as nn

log_probs = torch.randn(10, 256)
ctc_loss = nn.CTCLoss()
targets = torch.randint(1, 20, (15,), dtype=torch.long)
input_lengths = torch.full((10,), 256, dtype=torch.long)
target_lengths = torch.full((10,), 15, dtype=torch.long)

ctc_loss(log_probs, targets, input_lengths, target_lengths)