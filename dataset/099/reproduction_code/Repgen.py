import torch
import transformer_engine as te
import transformer_engine.pytorch as tep
from transformer_engine.common import recipe
import numpy as np

def reproduce_nan_loss_fp8():
    # Check if device supports FP8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, cannot test FP8 training")
        return False
    
    try:
        # Initialize a transformer layer with FP8 mixed precision
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.E4M3
        )
        
        # Create a simple transformer layer using Transformer Engine
        transformer_layer = tep.TransformerLayer(
            hidden_size=256,
            ffn_hidden_size=1024,
            num_attention_heads=8,
            fp8_recipe=fp8_recipe
        ).to(device)
        
        # Create a random input tensor
        batch_size = 8
        seq_len = 32
        hidden_size = 256
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Create a target tensor
        target = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(transformer_layer.parameters(), lr=1e-3)
        
        # Training loop
        num_iterations = 50
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = transformer_layer(input_tensor)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(output, target)
            
            print(f"Iteration {i}, Loss: {loss.item()}")
            
            # Check for NaN
            if torch.isnan(loss).any():
                print(f"✓ Bug reproduced: NaN loss detected at iteration {i}")
                return True
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        print("✗ Bug not reproduced: No NaN loss detected after training")
        return False
        
    except Exception as e:
        print(f"Error during reproduction: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    reproduce_nan_loss_fp8()