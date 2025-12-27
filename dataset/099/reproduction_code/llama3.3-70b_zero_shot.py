import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TransformerModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_ids = torch.randint(0, 100, size=(1, 10)).to(device)
attention_mask = torch.randint(0, 2, size=(1, 10)).to(device)
labels = torch.randint(0, 8, size=(1,)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for _ in range(10):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())