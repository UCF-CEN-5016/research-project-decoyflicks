import torch
from transformers import TransformerWrapper

# Define the transformer layer, setting only one of attn_num_mem_kv or attn_one_kv_head to True
transformer_layer = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=-1,  # Setting this to -1 indicates not using memory tokens
    attn_layers={
        'dim': 512,
        'depth': 4,
        'heads': 4,
        'rotary_pos_emb': True,
        'attn_flash': True,
        'attn_onnxable': True,
        'attn_num_mem_kv': -1,  # Using negative value to disable memory management
        'attn_one_kv_head': False   # Disabling one of the conflicting options
    }
)

# Input data
batch_size = 8
seq_length = 1024
x = torch.randint(0, 32, (batch_size, seq_length))
logits = transformer_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {logits.shape}")