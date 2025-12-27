import torch

class TransformerConfig:
    def __init__(self, num_tokens, max_seq_len, num_memory_tokens, attn_layers_config):
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.attn_layers_config = attn_layers_config

class DecoderConfig:
    def __init__(self, dim, depth, heads, rotary_pos_emb, attn_flash, attn_onnxable, attn_num_mem_kv, attn_one_kv_head):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.rotary_pos_emb = rotary_pos_emb
        self.attn_flash = attn_flash
        self.attn_onnxable = attn_onnxable
        self.attn_num_mem_kv = attn_num_mem_kv
        self.attn_one_kv_head = attn_one_kv_head

class TransformerWrapper(torch.nn.Module)
    def __init__(self, config):
        super(TransformerWrapper, self).__init__()
        self.num_tokens = config.num_tokens
        self.max_seq_len = config.max_seq_len
        self.num_memory_tokens = config.num_memory_tokens
        self.attn_layers = Decoder(config.attn_layers_config)

    def forward(self, x):
        return self.attn_layers(x)

config = TransformerConfig(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers_config=DecoderConfig(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    )
)

lm = TransformerWrapper(config)
x = torch.randint(0, 32, (8, 1024))
logits = lm(x)

print(x.shape, x.dtype)
print(logits.shape, logits.dtype)