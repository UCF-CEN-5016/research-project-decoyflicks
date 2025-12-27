import torch
from torch.audio import ForcedAlign

# Set up minimal environment
torch.manual_seed(42)  # for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add triggering conditions
audio_file = "audio.wav"
text_file = "text.txt"

# Wrap final code in a function to simulate the `align_and_segment.py` script
def reproduce_bug():
    # Load audio file and text transcription
    audio, _ = ForcedAlign.load_audio(audio_file)
    text = open(text_file, "r").read()

    # Trigger the bug by calling the forced alignment function
    try:
        log_probs = ForcedAlign(forced_alignment=True)(audio, text)
        print("Expected error: log_probs must be 3-D (batch_size, input length, num classes)")
    except RuntimeError as e:
        print(f"Error message: {e}")

reproduce_bug()