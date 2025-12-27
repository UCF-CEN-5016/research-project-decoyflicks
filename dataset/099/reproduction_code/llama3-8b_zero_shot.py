import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

input_ids = torch.tensor([[1, 2, 3]])
attention_mask = torch.tensor([[1, 1, 1]])

output = model(input_ids, attention_mask=attention_mask)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output.logits, torch.tensor([[2]]))

print(loss)