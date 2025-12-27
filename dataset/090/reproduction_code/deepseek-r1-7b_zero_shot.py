import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, z_dim=2, input_dim=784):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2_mean = nn.Linear(512, z_dim)
        self.fc2_logvar = nn.Linear(512, z_dim)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2_mean(h), self.fc2_logvar(h)

def train(model, optimizer, dataloader, loss_fn):
    model.train()
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.cuda() if torch.cuda.is_available() else data
        mu, logvar = model(data)
        # Calculate loss and backpropagate
    
    # The bug occurs here: to_logits.weight is not used in the forward pass of the model layers

model = Encoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if __name__ == '__main__':
    import dataloader  # Assuming some data loading code exists
    train(model, optimizer, dataloader, F.mse_loss)