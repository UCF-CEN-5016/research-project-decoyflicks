import torch
from continuous_transformer import ContinuousTransformerWrapper
from continuous_transformer.attn import Decoder

# Minimal reproduction setup
def reproduce_return_mems_bug():
    # Initialize the model with given parameters
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

    # Input tensor
    x = torch.randn(1, 1024, 512)

    # Mask tensor
    m = torch.randn(1, 1024) > 0

    # mems list with 6 tensors of shape (1, 100, 512)
    mems = [torch.zeros(1, 100, 512) for _ in range(6)]

    # Call with mems provided -> triggers bug
    try:
        logits, mems_out = net(x, mask=m, mems=mems, return_mems=True)
        print("Output logits shape:", logits.shape)
        print("Output mems shapes:", [mem.shape for mem in mems_out])
    except Exception as e:
        print("Error with mems provided:", e)

    # Call with mems=None -> works correctly
    try:
        logits, mems_out = net(x, mask=m, mems=None, return_mems=True)
        print("Output logits shape (mems=None):", logits.shape)
        print("Output mems shapes (mems=None):", [mem.shape for mem in mems_out])
    except Exception as e:
        print("Error with mems=None:", e)

if __name__ == "__main__":
    reproduce_return_mems_bug()