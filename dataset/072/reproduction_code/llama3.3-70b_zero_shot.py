import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedEngine, DeepSpeedConfig

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def forward(self, input_ids):
        return self.model(input_ids)

model = Model()
engine, _, _, _ = DeepSpeedEngine.initialize(args=DeepSpeedConfig({"train_batch_size": 16}), model=model, model_parameters=model.parameters())

input_ids = torch.randint(0, 100, size=(16, 10))
output = engine(input_ids)

print(engine)