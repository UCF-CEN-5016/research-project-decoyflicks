import torch
from x_transformers import TransformerWrapper

torch.manual_seed(0)

enc_seq_len = 258
dec_seq_len = 230
vocab_size = 100

model = TransformerWrapper(
    num_tokens = vocab_size,
    max_seq_len = enc_seq_len,
    attn_layers = dict(
        dim = 64,
        depth = 2,
        heads = 4,
        cross_attend = True,
    )
)

src = torch.randint(0, vocab_size, (1, enc_seq_len))
start_tokens = torch.randint(0, vocab_size, (1, dec_seq_len))

src_mask = torch.ones(1, enc_seq_len).bool()

model.generate(src, start_tokens, enc_seq_len, mask = src_mask)