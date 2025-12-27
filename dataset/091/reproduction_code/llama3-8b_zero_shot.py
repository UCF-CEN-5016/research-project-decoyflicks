import torch
from x_transformers import AutoModelForSeq2SeqLM, Seq2SeqEncoder, Seq2SeqDecoder

model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', return_dict=True)
encoder = model.get_encoder()
decoder = model.get_decoder()

start_tokens = torch.tensor([[0]])
src = torch.tensor([[1, 2], [3, 4], [5, 6]])  # encoding sequence (2x7)
encodings = src
mask = torch.tensor([[False], [True], [False]])  # mask for the encoding sequence

sample = model.generate(start_tokens=start_tokens,
                        max_length=7,
                        num_beams=1,
                        early_stopping=True,
                        enc_seq_len=len(src),
                        mask=mask)

print(sample)