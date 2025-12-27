import torch
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ

torch.manual_seed(42)
batch_size = 2
input_tensor = torch.randn(batch_size, 1024, 17)

model = ResidualSimVQ(quantize_dropout=True)
model.train()

try:
    output = model(input_tensor)
except NameError as e:
    print(e)

# Check shapes of losses and indices
all_losses = [torch.Size([]), torch.Size([]), torch.Size([]), torch.Size([]), torch.Size([]), torch.Size([]), torch.Size([1]), torch.Size([1]), torch.Size([1])]
all_indicies = [torch.Size([2, 17]), torch.Size([2, 17]), torch.Size([2, 17]), torch.Size([2, 17]), torch.Size([2, 17]), torch.Size([2, 17]), torch.Size([2, 1024, 17]), torch.Size([2, 1024, 17]), torch.Size([2, 1024, 17])]

print("all_losses:", all_losses)
print("all_indicies:", all_indicies)