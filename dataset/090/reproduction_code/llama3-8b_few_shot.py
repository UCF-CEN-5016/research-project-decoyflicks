import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EncoderDecoderModel

# Define a custom dataset and data loader
class CustomDataset(Dataset):
    def __init__(self):
        self.data = {"input_ids": [[1, 2, 3], [4, 5, 6]], "attention_mask": [[1, 1, 0], [1, 1, 1]]}

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data["input_ids"][idx]),
            "attention_mask": torch.tensor(self.data["attention_mask"][idx])
        }

dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=2)

# Define an encoder-decoder model
model = EncoderDecoderModel.from_pretrained("bert-base-uncased")

# Set the `to_logits` attribute of the encoder
model.encoder.to_logits.weight.requires_grad = False

# Train the model
for epoch in range(5):
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)

        # Backward pass and optimization
        loss = outputs.loss
        loss.backward()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.step()

print("Model weights updated:", model.encoder.to_logits.weight.requires_grad)