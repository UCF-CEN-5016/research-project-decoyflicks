import torch
from matplotlib import pyplot as plt
import tensorflow_models as tfm

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

X = torch.randn(32, 10)
y = torch.randn(32, 10)

# Set the backend to an interactive type after importing TensorFlow Models
plt.switch_backend('TkAgg')

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = nn.MSELoss()(outputs, y)
    loss.backward()
    optimizer.step()
    
    plt.figure()
    plt.plot(loss.item())
    plt.title(f'Loss at Epoch {epoch}')
    plt.show()