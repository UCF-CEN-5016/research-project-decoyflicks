import torch
import torchaudio

# Simulate audio file
audio_file = torch.randn(1, 16000)  # 1 second of audio

# Simulate model and emission generation
def generate_emissions(model, audio_file):
    emissions_arr = []
    chunk_size = 1000
    for i in range(0, len(audio_file[0]), chunk_size):
        chunk = audio_file[:, i:i+chunk_size]
        emission = torch.randn(1, chunk_size)  # Simulate emission
        emissions_arr.append(emission)
    
    # Simulate mismatched tensor sizes
    emissions_arr[1] = torch.randn(1, chunk_size + 150)  # Introduce size mismatch
    
    # This will cause the error
    emissions = torch.cat(emissions_arr, dim=1).squeeze()
    return emissions

# Run the problematic code
model = None  # Simulate model
audio_file = torch.randn(1, 16000)  # Simulate audio file
emissions = generate_emissions(model, audio_file)
print(emissions)