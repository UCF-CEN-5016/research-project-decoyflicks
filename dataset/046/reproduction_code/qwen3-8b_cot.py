import torch
import torchaudio

# 🧪 Step 1: Create a fake audio tensor (batch, channels, time)
audio = torch.randn(1, 1, 100)  # (batch=1, channels=1, time=100)

# 🧪 Step 2: Define a model that outputs 2D (batch, classes)
model = torch.nn.Linear(100, 10)  # Output shape: (batch, 10)

# 🧪 Step 3: Process the audio through the model
log_probs = model(audio.squeeze(1))  # (batch, classes) = (1, 10)

# 🧪 Step 4: Define a helper function to validate log_probs shape
def check_log_probs_shape(log_probs):
    if log_probs.dim() != 3:
        raise RuntimeError("log_probs must be 3-D (batch_size, input length, num classes)")

# 🧪 Step 5: Trigger the error by calling the validation function
try:
    check_log_probs_shape(log_probs)
except RuntimeError as e:
    print("Error:", e)