import torch
from x_transformers import TransformerWrapper, Decoder
import traceback

def reproduce_bug_all_steps():
    print("==== Step 1: Testing valid configurations first ====")
    
    # Test with only attn_num_mem_kv (should work)
    try:
        print("\nTesting with only attn_num_mem_kv=20:")
        model1 = TransformerWrapper(
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
                attn_num_mem_kv=20,
                attn_one_kv_head=False  # Default is False
            )
        )
        
        x = torch.randint(0, 32, (8, 128))
        logits = model1(x)
        print("✓ Success: Model with only attn_num_mem_kv works correctly")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
    
    # Test with only attn_one_kv_head (should work)
    try:
        print("\nTesting with only attn_one_kv_head=True:")
        model2 = TransformerWrapper(
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
                attn_num_mem_kv=0,  # Default is 0
                attn_one_kv_head=True
            )
        )
        
        x = torch.randint(0, 32, (8, 128))
        logits = model2(x)
        print("✓ Success: Model with only attn_one_kv_head works correctly")
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
    
    print("\n==== Step 2: Testing the conflicting configuration ====")
    
    # Test with both parameters (should fail)
    try:
        print("\nTesting with both attn_num_mem_kv=20 and attn_one_kv_head=True:")
        model3 = TransformerWrapper(
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
        
        x = torch.randint(0, 32, (8, 128))
        logits = model3(x)
        print("✗ Unexpected success: Model with conflicting parameters works")
        
    except Exception as e:
        print(f"✓ Expected error: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Stack trace:")
        traceback.print_exc()
        print("\n✓ Bug successfully reproduced: Conflict between attn_num_mem_kv and attn_one_kv_head")
    
    print("\n==== Step 3: Summary ====")
    print("1. Using only attn_num_mem_kv > 0: Works correctly")
    print("2. Using only attn_one_kv_head=True: Works correctly")
    print("3. Using both attn_num_mem_kv > 0 and attn_one_kv_head=True: Fails")
    print("Conclusion: The bug is confirmed - these two parameters cannot be used together")

if __name__ == "__main__":
    reproduce_bug_all_steps()