import torch

# Step 1: Create a dummy audio tensor with shape (batch_size, time_steps)
audio = torch.randn(1, 16000)  # batch_size=1, time_steps=16000

# Step 2: Create a dummy model that outputs 2D log_probs (batch_size, num_classes)
# This is a simple linear model instead of the forced aligner
model = torch.nn.Linear(16000, 10)  # num_classes=10
log_probs = model(audio)  # Shape: (batch_size, num_classes) = (1, 10)

# Step 3: Simulate the forced aligner's usage with correct input shape
batch_size = audio.size(0)
time_steps = log_probs.size(1)
num_classes = log_probs.size(2)

# Reshape log_probs to 3D tensor (batch_size, time_steps, num_classes)
log_probs_3d = log_probs.unsqueeze(1).expand(batch_size, time_steps, num_classes)

# Use the reshaped log_probs in the forced aligner
try:
    # Simulate the forced aligner's usage
    # Note: The actual forced aligner expects 3D log_probs (batch_size, time_steps, num_classes)
    # But here, we're using the reshaped 3D tensor
    # This is a simplified version to show the error
    forced_aligner(
        audio_tensor=audio,
        log_probs=log_probs_3d,
        text="test"
    )
except ValueError as e:
    print("Caught an error due to incorrect log_probs shape:", e)