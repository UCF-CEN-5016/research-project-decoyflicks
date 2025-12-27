import torch
from x_transformers import AutoregressiveWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = torch.randint(0, 100, (1, 258)).to(device)
start_tokens = torch.randint(0, 100, (1, 230)).to(device)
ENC_SEQ_LEN = 258
src_mask = torch.ones((1, 1, 258), device=device)

model = AutoregressiveWrapper(...)  # Initialize with appropriate parameters

try:
    sample = model.generate(src, ENC_SEQ_LEN, eos_token=None, prompt_lens=start_tokens)
except Exception as e:
    print(e)