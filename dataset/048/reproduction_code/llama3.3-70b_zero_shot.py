import torch
import numpy as np

def generate_emissions(model, audio_file):
    emissions_arr = []
    emissions_arr.append(torch.randn(1, 1649, 10))
    emissions_arr.append(torch.randn(1, 1799, 10))
    emissions = torch.cat(emissions_arr, dim=1).squeeze()
    return emissions

def main():
    model = torch.nn.Module()
    audio_file = "audio.wav"
    emissions = generate_emissions(model, audio_file)

if __name__ == "__main__":
    main()