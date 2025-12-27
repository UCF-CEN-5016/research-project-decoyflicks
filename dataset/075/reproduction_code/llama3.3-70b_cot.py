import deepspeed
from transformers import GPT2Model, GPT2Tokenizer

# Set up the model and tokenizer
model = GPT2Model.from_pretrained('gpt2-xl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

# Create a DeepSpeed configuration
ds_config = {
    'train_batch_size': 16,
    'train_micro_batch_size_per_gpu': 16,
    'steps_per_print': 1000,
    'wall_clock_breakdown': True
}

# Initialize the DeepSpeed model
model, _, _, _ = deepspeed.initialize(args=ds_config, model=model, model_parameters=model.parameters())

# This line should trigger the bug
try:
    # The 'token' keyword argument is not expected by ParquetConfig
    parquet_config = deepspeed.ParquetConfig(token='some_token')
except TypeError as e:
    print(f"Error: {e}")