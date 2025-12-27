import torch
import torch.nn.functional as F
import warnings

# Enable warnings to see the broadcasting warning
warnings.filterwarnings('always')

def minimal_reproduction():
    """
    Minimal reproduction of the commit loss calculation issue with mask
    """
    print("Reproducing the LFQ commit loss calculation issue with mask...")
    
    # Create a simplified version of the problem
    # The key issue is in LFQ.forward() when calculating commit_loss with a mask
    
    # Create a tensor representing the original_input in the LFQ.forward method
    batch_size = 2
    seq_len = 10
    dim = 4
    
    # Create original input and a mask
    original_input = torch.randn(batch_size, seq_len, dim)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    mask[:, 5:] = False  # Mask out second half
    
    # This simulates what happens in LFQ.forward()
    # When a mask is provided, only masked tokens are kept: original_input[mask]
    original_input_masked = original_input[mask]
    print(f"Original input shape: {original_input.shape}")
    print(f"Masked input shape: {original_input_masked.shape}")
    
    # Create quantized tensor (also masked)
    quantized_masked = torch.randn_like(original_input_masked)
    
    # The bug happens when commit_loss is calculated
    # The shapes don't match between original_input and quantized
    print("\nCalculating commit_loss using masked tensors (correct approach)...")
    commit_loss_correct = F.mse_loss(original_input_masked, quantized_masked)
    print(f"Correct commit_loss: {commit_loss_correct.item()}")
    
    # This simulates the bug - if shapes are mismatched during loss calculation
    # Create a quantized tensor for full sequence (not just masked tokens)
    quantized = torch.zeros_like(original_input)
    quantized[mask] = quantized_masked
    
    print("\nCalculating commit_loss with mismatched shapes (bug)...")
    try:
        # This would trigger a UserWarning about broadcasting
        commit_loss_bug = F.mse_loss(original_input_masked, quantized)
        print(f"Broadcasting happened, got commit_loss: {commit_loss_bug.item()}")
        print("\nThis is the bug - a shape mismatch warning during broadcasting!")
        print(f"original_input_masked shape: {original_input_masked.shape}")
        print(f"quantized shape: {quantized.shape}")
    except RuntimeError as e:
        print(f"Error: {e}")
    
    print("\nThe issue in LFQ.forward() is that when using a mask:")
    print("1. original_input becomes original_input[mask] with shape [num_masked_tokens, dim]")
    print("2. But commit_loss is calculated without ensuring shapes match")
    print("3. This leads to incorrect broadcasting and the warning")

if __name__ == "__main__":
    minimal_reproduction()