import torch
import deepspeed
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Setup environment
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Initialize DeepSpeed engine
config = {
    "train_batch_size": 1,
    "train_micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    }
}

# 3. Trigger the bug by accessing model.model
with deepspeed.init_deepspeed(config):
    model_engine = deepspeed.initialize(model=model, config=config)
    # This line will raise the error
    print_throughput(model_engine.model, ...)  # ❌ Incorrect access

# Correct access to the original model
print_throughput(model_engine.module, ...)  # ✅ Correct access