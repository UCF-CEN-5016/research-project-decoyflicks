from transformers import GPT2LMHeadModel
from datasets import load_dataset
import deepspeed

def setup_model_and_dataset():
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return model, dataset

def configure_deepspeed_trainer(model, train_dataset):
    deepspeed_config = {
        "autotuning": {
            "enabled": True,
            "fast": True  # Tuning micro batch size only
        }
    }
    trainer = deepspeed.init_trainer(
        model=model,
        train_dataset=train_dataset["train"],
        args=deepspeed.DeepSpeedConfig(deepspeed_config)
    )
    return trainer

def train_model(trainer):
    trainer.train()

if __name__ == "__main__":
    model, dataset = setup_model_and_dataset()
    trainer = configure_deepspeed_trainer(model, dataset)
    train_model(trainer)