import torchaudio
from torchaudio.pipelines import FORCE_ALIGNMENT

# Load audio and text data
audio_file = "audio.wav"
text_file = "text.txt"

# Set up forced alignment pipeline with UDM language model
pipeline = FORCE_ALIGNMENT(
    lang="udm",
    uroman_path="uroman/bin",
)

# Run forced alignment with error-prone configuration
try:
    results = pipeline.run(audio_file, text_file)
except RuntimeError as e:
    print(f"Error: {e}")