import torch
import torchaudio

# Simulate the issue by creating incorrect shaped log_probs
def reproduce_bug():
    # This would be the output from some acoustic model
    # Incorrect: 2D tensor instead of 3D
    log_probs = torch.randn(100, 30)  # (time_steps, num_classes)
    
    # This is what the forced alignment expects
    # Correct shape should be: (batch_size, time_steps, num_classes)
    
    # Try to perform alignment (this would fail)
    try:
        transcript = "test words for alignment".split()
        torchaudio.functional.forced_align(log_probs, transcript, log_probs.size(-1))
    except RuntimeError as e:
        print(f"Error reproduced: {e}")
        print("This happens when log_probs isn't 3D (batch, time, classes)")
        print("Solution: Add batch dimension with unsqueeze(0)")

# Show the fix
def demonstrate_fix():
    log_probs = torch.randn(100, 30)  # Original 2D tensor
    fixed_log_probs = log_probs.unsqueeze(0)  # Add batch dimension
    
    print(f"Original shape: {log_probs.shape}")
    print(f"Fixed shape: {fixed_log_probs.shape}")
    
    # This would work now
    transcript = "test words for alignment".split()
    alignment = torchaudio.functional.forced_align(fixed_log_probs, transcript, fixed_log_probs.size(-1))
    print("Alignment succeeded with proper shape!")

if __name__ == "__main__":
    print("Reproducing the bug...")
    reproduce_bug()
    
    print("\nDemonstrating the fix...")
    demonstrate_fix()