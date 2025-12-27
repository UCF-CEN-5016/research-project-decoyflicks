import torch
import torchaudio
from torchaudio.models import wav2letter

# Dummy audio waveform (1 second of silence at 16kHz)
waveform = torch.zeros(16000).unsqueeze(0)  # Shape: (1, 16000)

# Instantiate wav2letter model (for example)
model = wav2letter.model()

# Forward pass without batching dimension in output or incorrect shape manipulation
log_probs = model(waveform)  # Suppose this returns (input_length, num_classes)

# Intentionally remove batch dimension to simulate the bug
log_probs = log_probs.squeeze(0)  # Now shape is (input_length, num_classes), 2D instead of 3D

# Simulate forced alignment expecting 3D log_probs
def forced_align(log_probs):
    if log_probs.dim() != 3:
        raise RuntimeError("log_probs must be 3-D (batch_size, input length, num classes)")
    # Alignment logic here (omitted)

# This should raise the RuntimeError
forced_align(log_probs)