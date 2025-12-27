import torch
import torchaudio
from fairseq.models import Wav2VecModel


def compute_emissions(wav_model, audio_path):
    """Generate emission tensor by concatenating two random tensors."""
    parts = [torch.randn(1, 1648), torch.randn(1, 1799)]
    return torch.cat(parts, dim=1).squeeze()


def resolve_alignments(wav_model, audio_path):
    """Return emissions and a fixed stride value."""
    return compute_emissions(wav_model, audio_path), 5


def main(args=None):
    wav_model = Wav2VecModel()
    emissions, stride = resolve_alignments(wav_model, "audio.wav")
    # Print a freshly generated emission tensor (keeps original behavior)
    print(torch.cat([torch.randn(1, 1648), torch.randn(1, 1799)], dim=1).squeeze())


if __name__ == "__main__":
    main(None)