import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set seed
torch.manual_seed(0)

# Define batch size and number of features
batch_size = 100
n_features = 32

# Generate synthetic data
X, y = make_classification(n_samples=10000, n_features=n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_torch = torch.from_numpy(X_scaled).to('cuda', dtype=torch.float16)
y_torch = torch.from_numpy(y.astype(np.int64)).to('cuda')

# Define model
class TestMod(torch.nn.Module):
    def __init__(self):
        super(TestMod, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model_fp32 = TestMod().to('cpu')
model_fp16 = TestMod().half().to('cuda')

# Define optimizer
optimizer = FusedLAMBAMP(model_fp16.parameters(), lr=0.01)

# Create DataLoader
dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and gradient scaler
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(20):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model_fp16(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader.dataset)}")

# Check parameter differences
def print_param_diff(model_fp32, model_fp16):
    for name, param in model_fp32.named_parameters():
        if param.requires_grad:
            fp32 = param.data.cpu().numpy()
            fp16 = model_fp16.state_dict()[name].data.cpu().numpy()
            diff = np.abs(fp32 - fp16).mean()
            print(f"{name}: {diff}")

print_param_diff(model_fp32, model_fp16)

# Load optimizer state dictionary and check again
optimizer.load_state_dict(torch.load('optimizer.pth'))
print_param_diff(model_fp32, model_fp16)

# Assert error
try:
    with torch.no_grad():
        x = torch.randn(10, n_features).to('cuda', dtype=torch.float16)
        F.gelu(x, approximate='tanh_approximation')
except TypeError as e:
    assert str(e) == "gelu(): argument 'approximate' must be str, not bool"