import torch
from x_transformers import AutoregressiveWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = torch.randint(0, 100, (1, 258)).to(device)
start_tokens = torch.randint(0, 100, (1, 230)).to(device)
ENC_SEQ_LEN = 258
src_mask = torch.ones((1, 1, 258, 258)).to(device)

model = AutoregressiveWrapper(...)  # Instantiate with appropriate parameters

try:
    sample = model.generate(src, ENC_SEQ_LEN, eos_token=None, temperature=1.0, prompt_lens=None, filter_logits_fn='top_k', restrict_to_max_seq_len=True, amateur_model=None, filter_kwargs={}, contrastive_decode_kwargs={}, cache_kv=True)
except Exception as e:
    print(e)