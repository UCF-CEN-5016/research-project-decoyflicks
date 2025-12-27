import torch
from x_transformers import AutoregressiveWrapper

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a source tensor with a length of 258
src = torch.randint(0, 100, (1, 258)).to(device)

# Create start tokens with a length of 230, which is shorter than src
start_tokens = torch.randint(0, 100, (1, 230)).to(device)

# Define the expected encoding sequence length
ENC_SEQ_LEN = 258

# Create a source mask for the full length of the source tensor
src_mask = torch.ones((1, 1, 258, 258)).to(device)

# Initialize the model (replace 'your_model_here' with the actual model)
# Note: This line is necessary for the code to run, but the actual model should be defined elsewhere.
model = AutoregressiveWrapper(net=None).to(device)  # Placeholder for the actual model

# Attempt to generate a sample using the model
try:
    sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask=src_mask)
except Exception as e:
    # Print the error message and shapes of the tensors for debugging
    print(e)
    print(src.shape, start_tokens.shape, src_mask.shape)