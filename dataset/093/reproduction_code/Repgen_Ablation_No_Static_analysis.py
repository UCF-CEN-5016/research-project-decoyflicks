import torch
import inspect
from x_transformers import TransformerWrapper, Decoder, Attention

def analyze_source_code():
    """
    Analyze the source code to identify why attn_num_mem_kv and attn_one_kv_head conflict
    """
    print("Analyzing Attention class for conflicts between attn_num_mem_kv and attn_one_kv_head\n")
    
    # Inspect the Attention class to find the conflict
    try:
        attention_init_src = inspect.getsource(Attention.__init__)
        print("Attention.__init__ source code excerpt:")
        print("-" * 50)
        
        # Print lines of code that are most relevant to the bug
        for line in attention_init_src.split('\n'):
            if "one_kv_head" in line or "num_mem_kv" in line:
                print(line.strip())
        
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"Could not inspect Attention.__init__: {e}\n")
    
    # Create a minimal model to analyze the parameters
    print("Creating models with various configurations to analyze parameters...\n")
    
    # Standard model (no conflict)
    model_standard = TransformerWrapper(
        num_tokens=32,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=64,
            depth=2,
            heads=4
        )
    )
    
    # Model with only attn_num_mem_kv
    model_mem_kv = TransformerWrapper(
        num_tokens=32,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=64,
            depth=2,
            heads=4,
            attn_num_mem_kv=20
        )
    )
    
    # Model with only attn_one_kv_head
    model_one_kv = TransformerWrapper(
        num_tokens=32,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=64,
            depth=2,
            heads=4,
            attn_one_kv_head=True
        )
    )
    
    # Try to create model with both (will likely fail)
    try:
        model_conflict = TransformerWrapper(
            num_tokens=32,
            max_seq_len=512,
            attn_layers=Decoder(
                dim=64,
                depth=2,
                heads=4,
                attn_num_mem_kv=20,
                attn_one_kv_head=True
            )
        )
        has_conflict_model = True
    except Exception as e:
        has_conflict_model = False
        conflict_error = str(e)
    
    # Analyze module structure
    print("Parameter analysis:")
    
    # Find attention modules in standard model
    attn_modules_standard = [m for n, m in model_standard.named_modules() if isinstance(m, Attention)]
    
    # Find attention modules in mem_kv model
    attn_modules_mem_kv = [m for n, m in model_mem_kv.named_modules() if isinstance(m, Attention)]
    
    # Find attention modules in one_kv model
    attn_modules_one_kv = [m for n, m in model_one_kv.named_modules() if isinstance(m, Attention)]
    
    # Compare relevant attributes
    if attn_modules_standard and attn_modules_mem_kv and attn_modules_one_kv:
        attn_std = attn_modules_standard[0]
        attn_mem = attn_modules_mem_kv[0]
        attn_one = attn_modules_one_kv[0]
        
        print("\nKey attention parameters that may conflict:")
        print(f"Standard model - heads: {getattr(attn_std, 'heads', 'N/A')}, kv_heads: {getattr(attn_std, 'kv_heads', 'N/A')}")
        print(f"With attn_num_mem_kv - heads: {getattr(attn_mem, 'heads', 'N/A')}, kv_heads: {getattr(attn_mem, 'kv_heads', 'N/A')}")
        print(f"With attn_one_kv_head - heads: {getattr(attn_one, 'heads', 'N/A')}, kv_heads: {getattr(attn_one, 'kv_heads', 'N/A')}")
        
        # Check for memory tokens
        has_mem_std = hasattr(attn_std, 'mem_k') and attn_std.mem_k is not None
        has_mem_mem = hasattr(attn_mem, 'mem_k') and attn_mem.mem_k is not None
        has_mem_one = hasattr(attn_one, 'mem_k') and attn_one.mem_k is not None
        
        print(f"\nMemory tokens:")
        print(f"Standard model has memory tokens: {has_mem_std}")
        print(f"With attn_num_mem_kv has memory tokens: {has_mem_mem}")
        print(f"With attn_one_kv_head has memory tokens: {has_mem_one}")
    
    # Try to reproduce the bug
    print("\nAttempting to reproduce the bug...")
    try:
        model = TransformerWrapper(
            num_tokens=32,
            max_seq_len=512,
            num_memory_tokens=20,
            attn_layers=Decoder(
                dim=64,
                depth=2,
                heads=4,
                attn_num_mem_kv=20,
                attn_one_kv_head=True
            )
        )
        x = torch.randint(0, 32, (2, 16))
        logits = model(x)
        print("✗ Bug not reproduced: Model initialized and ran successfully")
    except Exception as e:
        print(f"✓ Bug reproduced: {type(e).__name__}: {e}")
    
    # Analysis conclusion
    print("\nAnalysis conclusion:")
    if not has_conflict_model:
        print(f"The model with both attn_num_mem_kv and attn_one_kv_head failed with error: {conflict_error}")
        print("The likely reason for this conflict is that:")
        print("- attn_num_mem_kv > 0 adds learnable memory key/value pairs to each attention layer")
        print("- attn_one_kv_head=True forces all heads to share the same key/value projections")
        print("These features may be incompatible in their implementation, causing shape mismatches or other errors")
    else:
        print("Surprisingly, the model with both parameters initialized without error in this analysis")
        print("This may indicate that the bug is environment-specific or has been fixed in the version being used")

if __name__ == "__main__":
    analyze_source_code()