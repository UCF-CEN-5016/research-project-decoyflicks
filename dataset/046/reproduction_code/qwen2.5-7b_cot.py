import torch
import torch.nn as nn

def create_fake_audio_tensor(batch_size, channels, time):
    return torch.randn(batch_size, channels, time)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.squeeze(1)
        return self.linear(x)

def check_log_probs_shape(log_probs):
    if log_probs.dim() != 2:
        raise RuntimeError("log_probs must be 2-D (batch_size, num_classes)")

# 🧪 Step 1: Create a fake audio tensor (batch, channels, time)
audio = create_fake_audio_tensor(1, 1, 100)  # (batch=1, channels=1, time=100)

# 🧪 Step 2: Define a model that outputs 2D (batch, classes)
model = Model(100, 10)  # Output shape: (batch, 10)

# 🧪 Step 3: Process the audio through the model
log_probs = model(audio)

# 🧪 Step 4: Trigger the error by calling the validation function
try:
    check_log_probs_shape(log_probs)
except RuntimeError as e:
    print("Error:", e)