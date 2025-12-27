import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create a DeepSpeed engine
engine, _, _, _ = deepspeed.initialize(args={"train_batch_size": 16}, model=model)

# Attempt to access the model attribute
try:
    print(engine.model)
except AttributeError as e:
    print(f"Error: {e}")

# Correct way to access the model
print(engine.module)