import torch
from x_transformers import AutoregressiveWrapper

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create source tensor with random integers, simulating an encoding sequence
src = torch.randint(0, 100, (1, 258)).to(device)

# Create start tokens tensor with random integers, simulating a shorter decoding sequence
start_tokens = torch.randint(0, 100, (1, 230)).to(device)

# Define the length of the encoding sequence
ENC_SEQ_LEN = 258

# Create a source mask for the encoding sequence
src_mask = torch.ones((1, 1, 258, 258)).to(device)

# Placeholder for the model; replace 'your_model_here' with the actual model instance
# This is necessary to avoid the undefined variable error
your_model_here = None  # Replace with actual model initialization

# Wrap the model with AutoregressiveWrapper
model = AutoregressiveWrapper(net=your_model_here).to(device)

# Attempt to generate a sample from the model
try:
    sample = model.generate(
        src,
        ENC_SEQ_LEN,
        eos_token=None,
        prompt_lens=start_tokens.shape[1],
        filter_logits_fn='top_k',
        restrict_to_max_seq_len=True
    )
except Exception as e:
    # Print the exception to observe the bug
    print(e)