import argparse
import datasets
import deepspeed
from learning_rates import AnnealingLR
import logging
import random
import time
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, from_pretrained, get_scheduler

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--eval_file', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

# Initialize DeepSpeed
deepspeed.init_distributed()
ds_config = deepspeed.add_config_arguments(args)
model = from_pretrained(args.model_type).to(torch.device("cuda"))

# Load dataset
train_dataset = datasets.load_dataset('json', data_files=args.train_file)['train']
eval_dataset = datasets.load_dataset('json', data_files=args.eval_file)['validation']

# Create data loaders
train_sampler = torch.utils.data.RandomSampler(train_dataset)
eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_epochs)

# Convert model to random LTD (if specified in args)
if hasattr(args, 'random_ltd') and args.random_ltd:
    model = deepspeed.convert_to_random_ltd(model, total_layers=12, start_layer=6)

# Training loop
model.train()
for epoch in range(args.num_epochs):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        model.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Evaluate perplexity on the validation set
    model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        for batch in eval_dataloader:
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
        avg_eval_loss = eval_loss / len(eval_dataloader)
        perplexity = math.exp(avg_eval_loss)

    print(f"Epoch {epoch+1}: Perplexity = {perplexity:.4f}")

# Save the model
if torch.distributed.get_rank() == 0:
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(save_without_random_ltd(model_to_save), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)