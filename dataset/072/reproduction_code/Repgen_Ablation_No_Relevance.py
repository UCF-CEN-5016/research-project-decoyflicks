import argparse
from datasets import load_dataset
import deepspeed
from learning_rates import AnnealingLR
import logging
import random
import time
import torch
from transformers import AdamW, GPT2Block, from_pretrained

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="output")
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and tokenize the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = from_pretrained("gpt2")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(
    tokenized_datasets["train"],
    batch_size=args.batch_size,
    shuffle=True,
)

# Initialize the model and learning rate scheduler
model = from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
scheduler = AnnealingLR(model, warmup_steps=len(train_loader) * 1)

# Configure DeepSpeed
deepspeed.init_distributed()
model = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={"train_batch_size": args.batch_size}
)[0]

# Define evaluation function
def evaluate_model():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in train_loader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Train the model
for epoch in range(args.num_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss
        model.backward(loss)
        scheduler.step()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            model.step()

    # Evaluate the model periodically
    if (epoch + 1) % 1 == 0:
        avg_loss = evaluate_model()
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Save the trained model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

logger.info("Training complete, model saved.")