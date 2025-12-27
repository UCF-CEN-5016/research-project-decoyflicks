import torch
import torch.nn as nn
import torch.optim as optim

# Minimal Encoder with a parameter that is NOT used in forward
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Parameter that is NOT used in forward
        self.to_logits = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Use only self.linear, not self.to_logits
        return torch.relu(self.linear(x))

# Decoder that uses encoder output
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Full model
class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        enc_out = self.encoder(x)  # encoder.to_logits.weight is NOT used here
        dec_out = self.decoder(enc_out)
        return dec_out

def main():
    torch.manual_seed(0)

    # Hyperparameters
    input_dim = 5
    hidden_dim = 10
    output_dim = 2
    batch_size = 3

    model = EncoderDecoder(input_dim, hidden_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # Dummy data
    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size, output_dim)

    # Save initial weight for to_logits.weight
    initial_weight = model.encoder.to_logits.weight.data.clone()

    # Forward pass
    output = model(x)
    loss = loss_fn(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if encoder.to_logits.weight updated
    updated_weight = model.encoder.to_logits.weight.data

    print("Encoder to_logits.weight requires grad:", model.encoder.to_logits.weight.requires_grad)
    print("Encoder to_logits.weight grad:", model.encoder.to_logits.weight.grad)
    print("Weight changed:", not torch.allclose(initial_weight, updated_weight))

if __name__ == "__main__":
    main()