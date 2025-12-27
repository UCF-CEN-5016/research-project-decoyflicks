import torch

# Minimal code to reproduce the issue
emissions_arr = [
    torch.randn(1, 1649),
    torch.randn(1, 1799)
]

model = None  # Not used in this example
audio_file = "audio.wav"  # Not used in this example

def generate_emissions(model, audio_file):
    return torch.cat(emissions_arr, dim=1).squeeze()

emissions = generate_emissions(model, audio_file)

print("Emissions shape:", emissions.shape)