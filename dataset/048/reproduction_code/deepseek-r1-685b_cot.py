import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

# Simulate the error conditions
def generate_emissions(audio_file):
    # Mock model that returns emissions of inconsistent sizes
    model = Wav2VecModel.from_pretrained("facebook/mms-1b-all")
    model.eval()
    
    # Simulate processing audio in chunks with inconsistent sizes
    wav, sr = torchaudio.load(audio_file)
    emissions_arr = []
    
    # First chunk processing (returns 1649 frames)
    emissions_arr.append(torch.randn(1, 1649, 1024))
    
    # Second chunk processing (returns 1799 frames - mismatch)
    emissions_arr.append(torch.randn(1, 1799, 1024))
    
    # This will trigger the same error as in the bug report
    emissions = torch.cat(emissions_arr, dim=1).squeeze()
    return emissions, 2  # stride

# Main alignment function that would trigger the error
def get_alignments(model, audio_file, text):
    emissions, stride = generate_emissions(audio_file)
    # Rest of alignment logic would go here...

# Mock setup to reproduce
if __name__ == "__main__":
    class Args:
        audio_filepath = "audio.wav"
        text_filepath = "text.txt"
        lang = "ful"
        outdir = "output"
        uroman = "uroman/bin"
    
    try:
        get_alignments(None, Args.audio_filepath, "sample text")
    except RuntimeError as e:
        print(f"Reproduced error: {e}")