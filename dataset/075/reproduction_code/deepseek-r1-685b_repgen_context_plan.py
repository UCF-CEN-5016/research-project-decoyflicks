from transformers import GPT2LMHeadModel
from datasets import load_dataset
import deepspeed

def setup_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    return model

def load_training_dataset():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset["train"]

def initialize_trainer(model, train_dataset):
    deepspeed_config = {
        "autotuning": {
            "enabled": True,
            "fast": True  # Tuning micro batch size only
        }
    }
    trainer = deepspeed.init_trainer(model=model, train_dataset=train_dataset, args=deepspeed.DeepSpeedConfig(deepspeed_config))
    return trainer

def train_model(trainer):
    trainer.train()

if __name__ == "__main__":
    model = setup_model()
    train_dataset = load_training_dataset()
    trainer = initialize_trainer(model, train_dataset)
    train_model(trainer)