import torch
from xtransformers import AutoModelForSeq2SeqLM, AlibiAttention

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

attn_flash = AlibiAttention(input_ids, attention_mask, num_heads=8)

custom_position = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
attn_flash.attn_bias = custom_position

print(attn_flash)