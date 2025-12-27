import torch
import deepspeed
from transformers import AutoModelForCausalLM

# Initialize model with DeepSpeed
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

# Simulate training (would normally have actual training loop here)
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
start_time.record()
end_time.record()

# This will raise the AttributeError
try:
    print(f"Throughput: {model_engine.model}")  # Wrong attribute access
except AttributeError as e:
    print(f"Error: {e}")
    print("Solution: Use 'module' instead of 'model'")
    print(f"Correct throughput: {model_engine.module}")  # Correct access