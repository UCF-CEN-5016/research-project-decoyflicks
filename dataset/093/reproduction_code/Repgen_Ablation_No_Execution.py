import torch
from x_transformers import TransformerWrapper, Decoder

def reproduce_bug():
    print("Initializing transformer with conflicting parameters...")
    
    try:
        # Create a transformer with the conflicting parameters
        model = TransformerWrapper(
            num_tokens=32,
            max_seq_len=1024,
            num_memory_tokens=20,
            attn_layers=Decoder(
                dim=512,
                depth=4,
                heads=4,
                rotary_pos_emb=True,
                attn_flash=True,
                attn_onnxable=True,
                attn_num_mem_kv=20,  # Conflict parameter 1
                attn_one_kv_head=True  # Conflict parameter 2
            )
        )
        
        print("Model initialized successfully (unexpected)")
        
        # Try to run a forward pass
        x = torch.randint(0, 32, (8, 128))
        print("Running forward pass...")
        
        logits = model(x)
        print("Forward pass completed successfully (unexpected)")
        
        return False  # Bug not reproduced
        
    except Exception as e:
        print(f"Error encountered: {type(e).__name__}: {e}")
        print("✓ Bug successfully reproduced")
        return True  # Bug reproduced

if __name__ == "__main__":
    reproduce_bug()