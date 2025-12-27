import torch
from torch import nn, optim

class DALLER(nn.Module):
    def __init__(self):
        super(DALLER, self).__init__()
        self.fc1 = nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, text, image_codes, mask=None):
        x = self.fc1(text)
        if mask is not None:
            x = x * mask
        return self.fc2(self.relu(x))

dalle_model = DALLER()

# ... some code to train dalle ...

def fit(model, opt, criterion, scheduler, train_x, train_y, epochs, batch_size, model_file, trainer):
    for epoch in range(epochs):
        # ...
        loss = trainer(dalle_model, train_x, train_y, None, criterion)
        loss.backward()
        losses.append(loss.item())

def train_dalle_batch(vae, train_data, _, idx, __):
    text, image_codes, mask = train_data
    loss = dalle(text[idx, ...], image_codes[idx, ...], mask=mask[idx, ...], return_loss=True)
    return loss

# Train DALL-E
dalle, loss_history = fit(dalle_model, opt, None, scheduler, (captions_array[train_idx, ...], all_image_codes[train_idx, ...], captions_mask[train_idx, ...]), None, 200, 256, "data/rainbow_dalle.model", train_dalle_batch)