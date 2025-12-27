import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# Set up FP8 training environment
torch.set_float32_matmul_precision('high')  # For modern GPUs

class SimpleTransformer(te.transformer.TransformerLayer):
    def __init__(self, hidden_size, num_heads):
        super().__init__(
            hidden_size=hidden_size,
            ffn_hidden_size=4*hidden_size,  # Standard 4x expansion
            num_attention_heads=num_heads,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            fp8=True,  # Enable FP8
            fp8_calibration=False,
            fp8_recipe=DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=32,
                amax_compute_algo='max',
            )
        )

def reproduce_nan_loss():
    # Model config
    hidden_size = 512
    num_heads = 8
    batch_size = 32
    seq_length = 128
    
    # Create model and optimizer
    model = SimpleTransformer(hidden_size, num_heads).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # FP8 training loop
    for step in range(100):
        # Synthetic data
        x = torch.randn(batch_size, seq_length, hidden_size).cuda()
        
        # Forward pass with FP8
        with te.fp8_autocast(enabled=True):
            out = model(x, attention_mask=None)
            loss = out.mean()  # Simple dummy loss
            
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN detected at step {step}")
            break
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (often needed for FP8 stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Step {step}, Loss: {loss.item()}")

if __name__ == "__main__":
    reproduce_nan_loss()