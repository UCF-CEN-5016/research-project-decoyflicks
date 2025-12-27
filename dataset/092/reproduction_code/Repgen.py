import torch
from x_transformers import TransformerWrapper, Decoder
import numpy as np

def reproduce_nan_bug_in_detail():
    print("=== Step 1: Model Setup ===")
    model = TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=Decoder(
            dim=1024,
            depth=24,
            heads=16,
            attn_dim_head=64,
            attn_flash=True,
            ff_no_bias=True,
            cross_attend=True,
        )
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("\n=== Step 2: Input Data Preparation ===")
    # Create input sequence
    seq_len = 20
    batch_size = 2
    i = torch.randint(0, 2048, (batch_size, seq_len))
    print(f"Input sequence shape: {i.shape}")
    
    # Create context
    context_len = 4
    context_dim = 1024
    context = torch.rand(batch_size, context_len, context_dim)
    print(f"Context shape: {context.shape}")
    
    # Create different context masks for testing
    context_mask_all_padding = torch.zeros(batch_size, context_len, dtype=torch.bool)
    context_mask_some_valid = torch.zeros(batch_size, context_len, dtype=torch.bool)
    context_mask_some_valid[:, 0] = True  # First token is valid
    context_mask_all_valid = torch.ones(batch_size, context_len, dtype=torch.bool)
    
    print("Context mask types created:")
    print(f"- All padding: {context_mask_all_padding}")
    print(f"- Some valid tokens: {context_mask_some_valid}")
    print(f"- All valid tokens: {context_mask_all_valid}")
    
    print("\n=== Step 3: Testing Different Scenarios ===")
    
    # Case 1: Without any context (baseline)
    print("\nCase 1: No context provided")
    try:
        output_no_context = model(i)
        has_nan = torch.isnan(output_no_context).any().item()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            nan_percentage = torch.isnan(output_no_context).float().mean().item() * 100
            print(f"NaN percentage: {nan_percentage:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    # Case 2: With context but no mask (should work)
    print("\nCase 2: With context, no mask")
    try:
        output_with_context = model(i, context=context)
        has_nan = torch.isnan(output_with_context).any().item()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            nan_percentage = torch.isnan(output_with_context).float().mean().item() * 100
            print(f"NaN percentage: {nan_percentage:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    # Case 3: With context and all-padding mask (should reproduce bug)
    print("\nCase 3: With context, all padding mask (bug case)")
    try:
        output_all_padding = model(i, context=context, context_mask=context_mask_all_padding)
        has_nan = torch.isnan(output_all_padding).any().item()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            nan_percentage = torch.isnan(output_all_padding).float().mean().item() * 100
            print(f"NaN percentage: {nan_percentage:.2f}%")
            print("✓ Bug reproduced: Output contains NaN values")
        else:
            print("✗ Bug not reproduced: Output does not contain NaN values")
    except Exception as e:
        print(f"Error: {e}")
    
    # Case 4: With context and some-valid mask (should work)
    print("\nCase 4: With context, some valid tokens in mask")
    try:
        output_some_valid = model(i, context=context, context_mask=context_mask_some_valid)
        has_nan = torch.isnan(output_some_valid).any().item()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            nan_percentage = torch.isnan(output_some_valid).float().mean().item() * 100
            print(f"NaN percentage: {nan_percentage:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    # Case 5: With context and all-valid mask (should work)
    print("\nCase 5: With context, all valid tokens in mask")
    try:
        output_all_valid = model(i, context=context, context_mask=context_mask_all_valid)
        has_nan = torch.isnan(output_all_valid).any().item()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            nan_percentage = torch.isnan(output_all_valid).float().mean().item() * 100
            print(f"NaN percentage: {nan_percentage:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Step 4: Summary ===")
    results = {
        "No context": torch.isnan(output_no_context).any().item() if 'output_no_context' in locals() else None,
        "With context, no mask": torch.isnan(output_with_context).any().item() if 'output_with_context' in locals() else None,
        "With context, all padding": torch.isnan(output_all_padding).any().item() if 'output_all_padding' in locals() else None,
        "With context, some valid": torch.isnan(output_some_valid).any().item() if 'output_some_valid' in locals() else None,
        "With context, all valid": torch.isnan(output_all_valid).any().item() if 'output_all_valid' in locals() else None
    }
    
    for case, has_nan in results.items():
        if has_nan is None:
            status = "❌ Error"
        elif has_nan:
            status = "❌ Has NaN"
        else:
            status = "✅ No NaN"
        print(f"{case}: {status}")
    
    print("\nConclusion:")
    if results["With context, all padding"]:
        print("The bug is reproduced: NaN values appear when the context is all padding.")
        print("This likely happens because the attention mechanism doesn't handle the case")
        print("where all context tokens are masked out, leading to division by zero or")
        print("other numerical instabilities in the attention computation.")
    else:
        print("The bug was not reproduced in this environment.")

if __name__ == "__main__":
    reproduce_nan_bug_in_detail()