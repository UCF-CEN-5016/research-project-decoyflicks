import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {"type": "AdamW", "params": {"lr": 5e-5}},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 1}
}

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]

# Simulate training step
input_ids = torch.randint(0, 100, (1, 10))
loss = ds_engine(input_ids, labels=input_ids).loss
ds_engine.backward(loss)
ds_engine.step()

# Attempt to access model attribute (will raise the error)
print(ds_engine.model)