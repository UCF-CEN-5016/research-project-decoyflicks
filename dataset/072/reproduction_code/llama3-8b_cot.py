import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedEngine

torch.cuda.is_available()  # Check if GPU is available

# Set the model and tokenizer (replace with your own)
model_name = "facebook/opt-30b-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a DeepSpeedEngine instance
engine = DeepSpeedEngine(model, arguments={"local_rank": 0})

# Train the model for one epoch (replace with your own training loop)
start_time = time.time()
for _ in range(1):  # Training for only one epoch
    engine.train()
end_time = time.time()

print(f"Training took {end_time - start_time} seconds.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedEngine

# Check if GPU is available
torch.cuda.is_available()

# Set the model and tokenizer (replace with your own)
model_name = "facebook/opt-30b-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a DeepSpeedEngine instance
engine = DeepSpeedEngine(model, arguments={"local_rank": 0})

# Train the model for one epoch (replace with your own training loop)
start_time = time.time()
for _ in range(1):  # Training for only one epoch
    engine.train()
end_time = time.time()

print(f"Training took {end_time - start_time} seconds.")