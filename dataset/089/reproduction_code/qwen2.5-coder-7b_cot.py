import torch
import torch.nn as nn
from x_transformers import XTransformer

DIM = 1024
SEQ_LEN = 512
DIM_HEAD = 64
QK_NORM = 3
KV_HEADS = 2
BATCH_SIZE = 8

def build_model(dim=DIM, seq_len=SEQ_LEN, qk_norm=QK_NORM, kv_heads=KV_HEADS, attend_mask=True):
    return XTransformer(dim=dim, seq_len=seq_len, qk_norm=qk_norm, kv_heads=kv_heads, attend_mask=attend_mask)

def run_inference(model, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, dim=DIM):
    input_tensor = torch.randn(batch_size, seq_len, dim)
    try:
        with torch.no_grad():
            return model(input_tensor)
    except Exception as exc:
        print(f"Error: {exc}")
        return None

if __name__ == "__main__":
    model = build_model()
    output = run_inference(model)