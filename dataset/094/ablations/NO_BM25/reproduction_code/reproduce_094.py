import torch
from palm_rlhf_pytorch import PaLM
from palm_rlhf_pytorch import ContinuousTransformerWrapper
from palm_rlhf_pytorch import Decoder

torch.manual_seed(42)

dim_out = 259
max_seq_len = 0
num_memory_tokens = 20
max_mem_len = 100

decoder_params = {
    'dim': 512,
    'depth': 6,
    'heads': 4,
    'rotary_pos_emb': True,
    'shift_tokens': 1,
    'attn_flash': True,
    'attn_onnxable': True,
    'use_rmsnorm': True,
    'sandwich_norm': True
}

decoder = Decoder(**decoder_params)
net = ContinuousTransformerWrapper(dim_out=dim_out, max_seq_len=max_seq_len, num_memory_tokens=num_memory_tokens, max_mem_len=max_mem_len, decoder=decoder)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([m.shape for m in mems])

logits, mems = net(x, mask=m, mems=None, return_mems=True)
print(logits.shape)