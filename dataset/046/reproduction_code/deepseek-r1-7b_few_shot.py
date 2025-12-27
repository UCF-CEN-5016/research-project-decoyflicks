import torch
from pyworld import readWave

# Simulate an audio file with n_frames and feature_dim
n_frames = 100
feature_dim = 32

def read_sim_audio(n_frames):
    return np.random.randn(n_frames).astype(np.float32)

waveform = read_sim_audio(n_frames)
x = torch.tensor(waveform, device='cuda').float().view(1, -1)  # (batch=1, time_steps=n_frames)

# Simple model expecting input of shape [batch=1, time_steps, features]
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, 32)
    
    def forward(self, x):
        # Reshape x to (time_steps, feature_dim) and add a singleton batch dimension
        return self.fc(x.view(-1, 1))  # Output shape: (n_frames, 32)

model = SimpleModel()
log_probs = model(x)

print(log_probs.shape)

import torch
from pyworld import readWave

# Simulate audio file with n_frames and feature_dim
n_frames = 100
feature_dim = 32

def read_sim_audio(n_frames):
    return np.random.randn(n_frames).astype(np.float32)

waveform = read_sim_audio(n_frames)
x = torch.tensor(waveform, device='cuda').float().view(1, -1)  # Shape: [batch=1, time_steps=n_frames]

# Simple model expecting input shape [batch=1, time_steps, features]
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, 32)
    
    def forward(self, x):
        # Reshape to match feature dimension
        batch_size, time_steps = x.size()
        return self.fc(x.view(-1, feature_dim))  # Output shape: [batch*time_steps, 32]

model = SimpleModel()
log_probs = model(x)

print(log_probs.shape)  # Outputs: torch.Size([n_frames, 32])