import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset for demonstration purposes
class MyDataset(Dataset):
    def __init__(self, freqs):
        self.freqs = freqs

    def __len__(self):
        return len(self.freqs)

    def __getitem__(self, idx):
        return {'freq': self.freqs[idx]}

# Create a dataset and data loader
freqs = torch.randn(100)
dataset = MyDataset(freqs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Define the model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.freq_cache = None

    def forward(self, freq):
        if self.freq_cache is None or len(freq) > len(self.freq_cache):
            self.freq_cache = freq
            return freq
        else:
            return self.freq_cache[:len(freq)]

# Initialize the model and optimizer
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(2):  # Trigger the bug on the second epoch
    for batch in dataloader:
        freq = batch['freq']
        outputs = model(freq)
        loss = torch.nn.functional.mse_loss(outputs, freq)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Batch {batch['freq'].shape[0]}: Loss = {loss.item()}")

print("Model output:", model(freq))