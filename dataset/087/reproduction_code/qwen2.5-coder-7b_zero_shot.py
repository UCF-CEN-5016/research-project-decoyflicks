import torch
from xtransformers import AutoModelForSeq2SeqLM, AlibiAttention

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

# Enable flash attention flag to reproduce the bug
attn_flash = AlibiAttention(input_ids, attention_mask, num_heads=8, attn_flash=True)

# Provide a 4D custom position tensor so the attention bias becomes 4D
custom_position = torch.randn(1, 1, 5, 5)
attn_flash.attn_bias = custom_position

print(attn_flash)