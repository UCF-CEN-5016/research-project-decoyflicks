import torch
from torch import nn

class DalleModel(nn.Module):
    def __init__(self):
        super(DalleModel, self).__init__()
        self.fc1 = nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, text, image_codes, mask=None, return_loss=False, **kwargs):
        x = self.fc1(text)
        if mask is not None:
            x = x * mask
        x = self.relu(x)
        out = self.fc2(x)
        # preserve compatibility with callers that pass return_loss
        return out

def train_dalle_batch(model, train_data, _, idx, __):
    text, image_codes, mask = train_data
    batch_text = text[idx, ...]
    batch_codes = image_codes[idx, ...]
    batch_mask = mask[idx, ...] if mask is not None else None
    loss = model(batch_text, batch_codes, mask=batch_mask, return_loss=True)
    return loss

def fit(model, opt, criterion, scheduler, train_x, train_y, epochs, batch_size, model_file, trainer):
    # train_x is expected to be a tuple: (texts, image_codes, masks)
    losses = []
    # determine number of samples from the first element of train_x
    if isinstance(train_x, (tuple, list)):
        n_samples = train_x[0].shape[0]
    else:
        n_samples = train_x.shape[0]

    for epoch in range(epochs):
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = slice(start, end)
            loss = trainer(model, train_x, train_y, batch_idx, criterion)
            loss.backward()
            losses.append(loss.item())
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

    return model, losses

dalle_model = DalleModel()

# Train DALL-E
dalle, loss_history = fit(
    dalle_model,
    opt,
    None,
    scheduler,
    (captions_array[train_idx, ...], all_image_codes[train_idx, ...], captions_mask[train_idx, ...]),
    None,
    200,
    256,
    "data/rainbow_dalle.model",
    train_dalle_batch
)