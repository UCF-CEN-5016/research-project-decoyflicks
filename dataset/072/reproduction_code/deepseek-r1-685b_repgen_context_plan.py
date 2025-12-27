import argparse
import time
import torch
import deepspeed
from transformers import AutoModel, AutoTokenizer

class TrainingConfig:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.batch_size = 8
        self.gradient_accumulation_steps = 1
        self.epochs = 1
        self.local_rank = -1

def print_throughput(model_instance, args, elapsed_time):
    samples_processed = args.batch_size * args.gradient_accumulation_steps
    print(f"Samples processed: {samples_processed}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Throughput: {samples_processed / elapsed_time:.2f} samples/second")

    param_count = sum(p.numel() for p in model_instance.parameters())
    print(f"Model parameter count: {param_count}")

def setup_and_run_deepspeed_training(config: TrainingConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=config.model_name)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--local_rank", type=int, default=config.local_rank)
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
    # This line is intentionally left as is to reproduce the AttributeError
    # as requested in the original problem description.
    print_throughput(model_engine.model, args, end - start)

    print("This line will not be reached due to the error above.")

if __name__ == "__main__":
    config = TrainingConfig()
    setup_and_run_deepspeed_training(config)
