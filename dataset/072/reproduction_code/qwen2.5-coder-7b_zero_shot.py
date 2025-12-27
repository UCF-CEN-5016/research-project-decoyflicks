import torch
from transformers import DeepSpeedEngine, AutoModelForCausalLM

model_name = "microsoft/deep-speed-chat"
model = AutoModelForCausalLM.from_pretrained(model_name)

engine = DeepSpeedEngine(model)
print(engine.model  # This line will raise an AttributeError