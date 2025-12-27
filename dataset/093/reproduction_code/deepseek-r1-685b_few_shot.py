import torch
from einops import rearrange
from x_transformers import TransformerWrapper, Decoder

Bug Type: Attention Configuration Conflict
Bug Description: Incompatible settings when both `attn_num_mem_kv` and `attn_one_kv_head` are enabled
Reproduction Code: