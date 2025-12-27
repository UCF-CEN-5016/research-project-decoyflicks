import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForQuestionAnswering

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
input_data = torch.randn(batch_size, 3, 224, 224).to(device)
target_data = torch.randint(0, 2, (batch_size, 2)).to(device)

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased').to(device)
model.eval()

try:
    output = F.gelu(input_data, approximate=True)
except TypeError as e:
    print(f"Error: {e}")