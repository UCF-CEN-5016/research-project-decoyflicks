import torch
from transformers import BertModel
import deepspeed

def initialize_model():
    model = BertModel.from_pretrained('bert-base-uncased')
    config = {
        "zero_optimization": {
            "stage": 3
        }
    }
    model, *_ = deepspeed.initialize(model, config)
    return model

# Initialize the model
model = initialize_model()

# This line will trigger the AttributeError
print(model.bert)