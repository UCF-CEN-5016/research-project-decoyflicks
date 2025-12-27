import torch

class BertIntermediate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(768, 3072)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # Using gelu without the approximate parameter (default behavior)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        return hidden_states

model = BertIntermediate()
inputs = torch.randn(2, 10, 768)
# This will work without error
outputs = model(inputs)
print(f"Output shape: {outputs.shape}")