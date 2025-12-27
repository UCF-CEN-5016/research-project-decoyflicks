import torch
from x_transformers import TransformerWrapper, Encoder, Decoder

ENC_SEQ_LEN = 230
DEC_SEQ_LEN = 258
DIM = 512

model = TransformerWrapper(
    num_tokens=256,
    max_seq_len=max(ENC_SEQ_LEN, DEC_SEQ_LEN),
    attn_layers=Decoder(
        dim=DIM,
        depth=6,
        cross_attend=True
    )
)

src = torch.randint(0, 256, (1, ENC_SEQ_LEN))
src_mask = torch.ones(1, ENC_SEQ_LEN).bool()
start_tokens = torch.randint(0, 256, (1, 1))

sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask=src_mask)