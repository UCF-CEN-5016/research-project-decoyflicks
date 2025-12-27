import torch
import torch.nn as nn
import torch.optim as optim

class SimpleBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 768)
        self.dense = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        hidden_states = self.dense(embeddings)
        # Will trigger the error during forward pass
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate=True)
        return self.classifier(hidden_states[:,0,:])

# Setup training
model = SimpleBertModel()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
input_ids = torch.randint(0, 30000, (8, 128))
labels = torch.randint(0, 2, (8,))

# This will raise the TypeError during training
model.train()
optimizer.zero_grad()
outputs = model(input_ids)
loss = nn.CrossEntropyLoss()(outputs, labels)
loss.backward()
optimizer.step()