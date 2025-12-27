import torch

def get_device():
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        print("MPS backend not available in this PyTorch version")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(3, 10).to(device)
print(model(x))