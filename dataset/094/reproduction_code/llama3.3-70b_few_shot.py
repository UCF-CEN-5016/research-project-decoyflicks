import torch

# Define a simple Decoder class for demonstration
class Decoder(torch.nn.Module):
    def __init__(self, dim, depth, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm):
        super(Decoder, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.rotary_pos_emb = rotary_pos_emb
        self.shift_tokens = shift_tokens
        self.attn_flash = attn_flash
        self.attn_onnxable = attn_onnxable
        self.use_rmsnorm = use_rmsnorm
        self.sandwich_norm = sandwich_norm

    def forward(self, x):
        return x

# Define a simple ContinuousTransformerWrapper class for demonstration
class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super(ContinuousTransformerWrapper, self).__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def forward(self, x, mask, mems=None, return_mems=False):
        if mems is not None:
            # Simulate the bug by not returning mems when return_mems is True
            return self.attn_layers(x), None
        else:
            return self.attn_layers(x), [torch.zeros(1, 100, 512) for _ in range(6)]

net = ContinuousTransformerWrapper(
    dim_out             = 259,
    max_seq_len         = 0,
    num_memory_tokens   = 20,
    max_mem_len         = 100,
    attn_layers = Decoder (
        dim             = 512,
        depth           = 6,
        heads           = 4,
        rotary_pos_emb  = True,
        shift_tokens    = 1,
        attn_flash      = True,
        attn_onnxable   = True,
        use_rmsnorm     = True,
        sandwich_norm   = True
    )
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]
logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([m.shape for m in mems])