import torch
import torchaudio
from torchaudio.models import forced_aligner

# Step 1: Create a dummy audio tensor with shape (batch_size, time_steps)
audio = torch.randn(1, 16000)  # batch_size=1, time_steps=16000

# Step 2: Create a dummy model that outputs 2D log_probs (batch_size, num_classes)
# This is a simple linear model instead of the forced aligner
model = torch.nn.Linear(16000, 10)  # num_classes=10
log_probs = model(audio)  # Shape: (batch_size, num_classes) = (1, 10)

# Step 3: Attempt to use this 2D log_probs in the forced aligner
# This is where the error will occur

# Simulate the forced aligner's usage
# Note: The actual forced aligner expects 3D log_probs (batch_size, time_steps, num_classes)
# But here, we're using the 2D tensor, which will cause a shape mismatch

# Example of what might be done in the forced aligner
# This is a simplified version to show the error
try:
    # Forcing the model to expect a 3D tensor (batch, time, class)
    # This will fail due to shape mismatch
    forced_aligner(
        audio_tensor=audio,
        log_probs=log_probs,
        text="test"
    )
except ValueError as e:
    print("Caught an error due to incorrect log_probs shape:", e)