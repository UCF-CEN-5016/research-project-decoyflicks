import torch
import torch.nn as nn
import torch.optim as optim

class MockDALLEModel(nn.Module):
    def __init__(self):
        super(MockDALLEModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, text, image_codes):
        return self.linear(text + image_codes)

def train_dalle_batch(model, data, idx):
    text, image_codes, mask = data
    loss = model(text[idx], image_codes[idx], mask=mask[idx], return_loss=True)
    return loss

def fit(model, optimizer, train_data, epochs, batch_size, model_file):
    text_data, image_data, mask_data = train_data
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        idx = torch.randint(0, len(text_data), (batch_size,))
        loss = train_dalle_batch(model, (text_data, image_data, mask_data), idx)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # Create dummy data
    text_data = torch.rand(100, 10)
    image_data = torch.rand(100, 10)
    mask_data = torch.rand(100, 10)
    
    model = MockDALLEModel()
    optimizer = optim.Adam(model.parameters())
    
    fit(model, optimizer, (text_data, image_data, mask_data), epochs=1, batch_size=10, model_file="model.pth")