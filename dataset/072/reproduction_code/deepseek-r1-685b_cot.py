import torch
from transformers import AutoModelForCausalLM
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

# Minimal setup to reproduce
model = AutoModelForCausalLM.from_pretrained("gpt2")
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {"type": "AdamW"},
}

# Initialize DeepSpeed engine
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# This will trigger the same error
try:
    print(model_engine.model)  # Wrong access
except AttributeError as e:
    print(f"Error occurred: {e}")  # Will show same error as in bug report

# Correct way to access the original model
print(model_engine.module)  # This works

print_throughput(model.model, args, end - start, ...)

print_throughput(model.module, args, end - start, ...)