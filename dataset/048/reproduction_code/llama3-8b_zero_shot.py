import torch
import torchaudio
from fairseq.models import Wav2VecModel

def generate_emissions(model, audio_file):
    emissions_arr = [torch.randn(1, 1648), torch.randn(1, 1799)]
    return torch.cat(emissions_arr, dim=1).squeeze()

def get_alignments(model, audio_file):
    return generate_emissions(model, audio_file), 5

def main(args):
    model = Wav2VecModel()
    _, stride = get_alignments(model, "audio.wav")
    print(torch.cat([torch.randn(1, 1648), torch.randn(1, 1799)], dim=1).squeeze())

if __name__ == "__main__":
    main(None)