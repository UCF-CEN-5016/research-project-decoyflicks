import os
import torch
import torchaudio
import sox
import json
import argparse

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dummy_audio():
    os.system("sox -n -r 16000 -c 1 audio.wav trim 0 5")

def create_text_file():
    with open("text.txt", "w") as f:
        f.write("This is a test transcription.")

def load_model_dict():
    # Dummy implementation for reproduction purposes
    model = torch.nn.Linear(10, 10)  # Replace with actual model loading logic
    dictionary = {"<blank>": 0}  # Dummy dictionary for reproduction
    return model, dictionary

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)
    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ
    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)
            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[:, int(SAMPLING_FREQ * input_start_time): int(SAMPLING_FREQ * (input_end_time))]
            model_outs = model(waveform_split)  # Adjusted to match model output
            emissions_ = model_outs.unsqueeze(0)  # Ensure emissions_ is 3D
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL
    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    return emissions

def main():
    create_dummy_audio()
    create_text_file()
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    tokens = ["<blank>"]  # Dummy tokens for reproduction
    emissions = generate_emissions(model, "audio.wav")
    print("Emissions shape:", emissions.shape)
    input_lengths = torch.tensor([emissions.shape[0]])  # Wrap in a list to create a tensor
    target_lengths = torch.tensor([len(tokens)])  # Wrap in a list to create a tensor
    blank = dictionary["<blank>"]
    path, _ = torchaudio.functional.forced_align(emissions, torch.tensor([0]), input_lengths, target_lengths, blank=blank)

if __name__ == "__main__":
    main()