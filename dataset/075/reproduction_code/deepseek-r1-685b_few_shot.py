from transformers import GPT2LMHeadModel
from datasets import load_dataset
import deepspeed

# Minimal setup to reproduce the error
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# This triggers the ParquetConfig error during autotuning
trainer = deepspeed.init_trainer(
    model=model,
    train_dataset=dataset["train"],
    args=deepspeed.DeepSpeedConfig({
        "autotuning": {
            "enabled": True,
            "fast": True  # Tuning micro batch size only
        }
    })
)

# Error occurs before this point
trainer.train()