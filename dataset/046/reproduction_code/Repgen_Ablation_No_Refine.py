import logging
import torch
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset

# Mock text romanization function
def mock_uroman(text):
    return text

# Load 'align_and_segment.py' script (assuming it's in the same directory)
from align_and_segment import CnnPredictor, Collator, train_epoch

# Set up logging
logger = logging.getLogger(__name__)

# Sample audio and text files
audio_filepath = "audio.wav"
text_filepath = "text.txt"

# Mock DataLoader setup with known data (batch_size=32, dynamic input_length, num_classes=1)
class SampleDataset(Dataset):
    def __init__(self):
        self.data = [
            ([1, 2, 3, 4, 5], [10, 20]),
            ([6, 7, 8, 9, 10], [30, 40])
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

sample_ds = SampleDataset()
collate_fn = Collator(padding_idx=0)
sample_dl = DataLoader(sample_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Instantiate CnnPredictor model
model = CnnPredictor(n_tokens=100, emb_dim=64, channels=128, kernel=3, output_dim=1, dropout=0.5, n_layers=2)

# Perform forward pass
log_probs = model(sample_dl.dataset[0][0])

# Assertions for log_probs shape and contents
assert len(log_probs.shape) == 3, f"Expected log_probs to be 3-D but got {len(log_probs.shape)}"
assert log_probs.shape[1] > 0, "Input length should be greater than 0"
assert log_probs.shape[2] == model.output_dim, f"Number of classes should match output_dim: {model.output_dim}"
assert torch.isfinite(log_probs).all(), "log_probs should not contain NaN or infinity"

# Save and load checkpoint (optional)
checkpoint_path = "sample_checkpoint.ckpt"
torch.save(model.state_dict(), checkpoint_path)

loaded_model = CnnPredictor(n_tokens=100, emb_dim=64, channels=128, kernel=3, output_dim=1, dropout=0.5, n_layers=2)
loaded_model.load_state_dict(torch.load(checkpoint_path))
assert loaded_model.state_dict() == model.state_dict(), "Loaded model state_dict does not match original model"

# Re-run forward pass with the same input batch
reloaded_log_probs = loaded_model(sample_dl.dataset[0][0])
assert torch.equal(log_probs, reloaded_log_probs), "Forward pass results do not match after checkpoint load"