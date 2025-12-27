import torch
from continuous_transformer import ContinuousTransformerWrapper

# Initialize model with necessary parameters
dim_out = 259
max_seq_len = 0
num_memory_tokens = 20
max_mem_len = 100

attn_layers = Decoder(
    dim=512,
    depth=6,
    heads=4,
    rotary_pos_emb=True,
    shift_tokens=1,
    attn_flash=True,
    attn_onnxable=True,
    use_rmsnorm=True,
    sandwich_norm=True
)

# Create input tensors
x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(num_memory_tokens)]

# When mems is None, it works; when provided as a list, return_mems fails
net = ContinuousTransformerWrapper(
    dim_out=dim_out,
    max_seq_len=max_seq_len,
    num_memory_tokens=num_memory_tokens,
    max_mem_len=max_mem_len,
    attn_layers=attn_layers
)

# Failing case
try:
    logits, mems = net(x, mask=m, mems=mems, return_mems=True)
except Exception as e:
    print(f"Error when using provided mems: {e}")

# Successful case (when mems=None)
logits, _ = net(x, mask=m, mems=None, return_mems=True)
print("成功获取输出和记忆")