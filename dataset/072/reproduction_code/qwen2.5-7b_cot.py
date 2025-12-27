import torch
import deepspeed
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def setup_environment(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def initialize_deepspeed_engine(model, config):
    with deepspeed.init_deepspeed(config):
        model_engine = deepspeed.initialize(model=model, config=config)
    return model_engine

def trigger_bug(model_engine, print_throughput_function):
    # This line will raise the error
    print_throughput_function(model_engine.model, ...)  # ❌ Incorrect access

def correct_access_original_model(model_engine, print_throughput_function):
    print_throughput_function(model_engine.module, ...)  # ✅ Correct access

# 1. Setup environment
model_name = "gpt2"
tokenizer, model = setup_environment(model_name)

# 2. Initialize DeepSpeed engine
config = {
    "train_batch_size": 1,
    "train_micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    }
}

model_engine = initialize_deepspeed_engine(model, config)

# 3. Trigger the bug by accessing model.model
trigger_bug(model_engine, print_throughput)

# Correct access to the original model
correct_access_original_model(model_engine, print_throughput)