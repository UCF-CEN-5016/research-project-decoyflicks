import torch
from transformers import BertModel
import deepspeed

model = BertModel.from_pretrained('bert-base-uncased')
config = {
    "zero_optimization": {
        "stage": 3
    }
}
model, *_ = deepspeed.initialize(model, config)

# This line will trigger the AttributeError
print(model.model)