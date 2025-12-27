import torch.nn.functional as F

# Define a simple BERT model
class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.gelu = lambda x: F.gelu(x, approximate=True)  # Passes bool instead of str

    def forward(self, x):
        return self.gelu(x)

model = BertModel()
input_tensor = torch.randn(1, 100)
output = model(input_tensor)
print(output)