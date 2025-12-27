import torch
from torchaudio.functional import waveform_to_features

# Set up minimal environment
model_path = 'uroman.bin'
audio_file = 'audio.wav'
text_file = 'text.txt'

# Add triggering conditions
emissions_arr = [
    torch.randn(1, 1649),  # Expected size: 1649
    torch.randn(1, 1799)   # Actual size: 1799
]

def generate_emissions(model_path, audio_file):
    # Simulate the model and audio file
    features = waveform_to_features(audio_file)
    emissions = []
    for i in range(len(emissions_arr)):
        emission = torch.cat((features[i], emissions_arr[i]), dim=1).squeeze()
        emissions.append(emission)
    return emissions

def main():
    # Call the problematic function
    emissions = generate_emissions(model_path, audio_file)

if __name__ == '__main__':
    main()