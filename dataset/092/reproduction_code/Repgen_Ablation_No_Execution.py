import torch
from x_transformers import TransformerWrapper, Decoder

def reproduce_bug():
    print("Setting up the model...")
    model = TransformerWrapper(
        num_tokens=2049,
        max_seq_len=500,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=True,
        attn_layers=Decoder(
            dim=1024,
            depth=12,
            heads=8,
            cross_attend=True,
            attn_flash=True  # Using flash attention as mentioned in the bug report
        )
    )
    
    print("Creating input data...")
    # Create random input sequence
    i = torch.randint(0, 2048, (2, 20))
    
    # Create random context
    context = torch.rand(2, 4, 1024)
    
    # Create context mask where all values are False (all padding)
    context_mask = torch.zeros(2, 4, dtype=torch.bool)
    
    print("Running forward pass with all-padding context...")
    # Forward pass with context and mask indicating all context is padding
    output = model(i, context=context, context_mask=context_mask)
    
    # Check for NaN values
    has_nan = torch.isnan(output).any().item()
    
    if has_nan:
        print("✓ Bug reproduced: Output contains NaN values")
        print(f"NaN percentage: {torch.isnan(output).float().mean().item() * 100:.2f}%")
    else:
        print("✗ Bug not reproduced: Output does not contain NaN values")
    
    # Compare with a working case (without context_mask)
    print("\nRunning forward pass without context_mask...")
    output_no_mask = model(i, context=context)
    
    has_nan_no_mask = torch.isnan(output_no_mask).any().item()
    if has_nan_no_mask:
        print("Output without context_mask also contains NaN values")
    else:
        print("Output without context_mask does not contain NaN values")
    
    return output, output_no_mask

if __name__ == "__main__":
    output, output_no_mask = reproduce_bug()