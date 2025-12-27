import torch
import torchaudio

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulate audio and text data
audio_file = torch.randn(1, 1000)  # Simulated audio data
text_transcription = "This is a simulated text transcription."

# Define a simple model for demonstration
class AlignmentModel(torch.nn.Module):
    def __init__(self):
        super(AlignmentModel, self).__init__()
        self.fc = torch.nn.Linear(1000, 100)

    def forward(self, x):
        return self.fc(x)

# Initialize the model and move it to the device
model = AlignmentModel()
model.to(device)

# Simulate generation of emissions
def generate_emissions(model, audio_file):
    emissions_arr = []
    # Simulate processing audio in chunks
    chunk_size = 100
    for i in range(0, len(audio_file), chunk_size):
        chunk = audio_file[i:i+chunk_size]
        chunk = chunk.to(device)
        emission = model(chunk)
        emissions_arr.append(emission)
    
    # Attempt to concatenate emissions along dimension 1
    try:
        emissions = torch.cat(emissions_arr, dim=1).squeeze()
    except RuntimeError as e:
        print(f"Error: {e}")
        return None
    
    return emissions

# Trigger the bug
emissions = generate_emissions(model, audio_file)

if emissions is None:
    print("Bug reproduced: Sizes of tensors must match except in dimension 1.")
else:
    print("Emissions generated successfully.")