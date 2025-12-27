import torch
from fairseq import EnsembleModel, get_lm_scores

# Define model parameters
beam_size = 5
batch_size = 32

# Create dummy input tokens
input_tokens = torch.randint(0, 100, (batch_size, beam_size), dtype=torch.long)

# Prepare incremental state dictionary
incremental_states = {layer: None for layer in range(10)}

# Define candidate token indices
cand_tokens = torch.randint(0, 100, (batch_size * beam_size, 5), dtype=torch.long)

# Dummy input length
input_len = torch.tensor([batch_size] * batch_size, dtype=torch.long)

# Call get_lm_scores function
probs_next_wrd = get_lm_scores(model=EnsembleModel(), input_tokens=input_tokens, incremental_states=incremental_states, cand_tokens=cand_tokens, input_len=input_len, k=beam_size)

# Verify NaN values in probs_next_wrd tensor
assert torch.isnan(probs_next_wrd).any()

# Monitor GPU memory usage
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory Usage: {info.used / 1024**2} MB")

# Reorder source tokens
from fairseq.models.transformer import reorder_tokens, reorder_all_tokens

new_order = torch.randperm(batch_size)
input_tokens = reorder_tokens(input_tokens, new_order)

input_len = input_len[new_order]