import torch
import deepspeed
from dschat.utils.data import create_prompt_dataset

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define dataset and output path
dataset_name = 'Dahoas/rm-static'
output_path = './data'

# Create training and evaluation datasets
train_dataset, eval_dataset = create_prompt_dataset(
    local_rank=0,
    data_path=[dataset_name],
    data_split='10,0',
    output_path=output_path,
    train_phase=1,
    seed=42,
    tokenizer=None,
    max_seq_len=512
)

# Configuration for DeepSpeed
config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 10,
    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "none"
        },
        "offload_optimizer": {
            "device": "none"
        }
    }
}

# Initialize the model with DeepSpeed
model, _, _, _ = deepspeed.initialize(model=None, model_parameters=None, config_params=config)

# Training loop for one epoch
for epoch in range(1):
    # Assuming train function is defined elsewhere
    train(model, train_dataset)  # Note: 'train' function must be defined to avoid errors

    # Attempt to print throughput, which will raise the AttributeError
    print_throughput(model.module, None, None)  # Changed from model.model to model.module to avoid AttributeError