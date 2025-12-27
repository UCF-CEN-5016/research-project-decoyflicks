import torch
import torch.nn.functional as F
from torch import nn

class TransformerWrapper(nn.Module):
    def __init__(self, dim, depth, heads, rotary_pos_emb=False, attn_flash=True, attn_onnxable=False,
                 attn_num_mem_kv=0, attn_one_kv_head=False):
        super(TransformerWrapper, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=None,
            dropout=0.0,
            activation=F.relu,
            layer_norms_first=True,
            rotary_pos_emb=rotary_pos_emb,
            attn Flash=attn_flash,
            attn Onnxable=attn_onnxable,
            attn_num_mem_kv=attn_num_mem_kv,
            attn_one_kv_head=attn_one_kv_head)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

def run_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerWrapper(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    ).to(device)
    
    x = torch.randint(0, 32, (8, 1024)).to(device)
    try:
        logits = model(x)
    except Exception as e:
        print(f"Error: {e}")
        
run_model()