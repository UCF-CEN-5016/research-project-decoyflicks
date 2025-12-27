import argparse
import time
import torch
import deepspeed
from transformers import AutoModel, AutoTokenizer

def print_throughput(model, args, elapsed_time):
    samples_processed = args.batch_size * args.gradient_accumulation_steps
    print(f"Samples processed: {samples_processed}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Throughput: {samples_processed / elapsed_time:.2f} samples/second")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {param_count}")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

deepspeed.init_distributed()

model = AutoModel.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

dummy_input = tokenizer("Hello, this is a test sentence.", return_tensors="pt", padding=True, truncation=True)
dummy_input = {k: v.repeat(args.batch_size, 1) for k, v in dummy_input.items()}

ds_config = {
    "train_batch_size": args.batch_size,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 1
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

model_engine.train()
start = time.time()

for _ in range(args.epochs):
    outputs = model_engine(**dummy_input)
    loss = outputs.last_hidden_state.mean()

    model_engine.backward(loss)
    model_engine.step()

end = time.time()

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.4f}")

print("Attempting to print throughput...")
print_throughput(model_engine.model, args, end - start)

print("This line will not be reached due to the error above.")
