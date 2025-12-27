import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import DeepSpeedEngine, DeepSpeedConfig

def main():
    model_name = "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "steps_per_print": 1000,
        "wall_clock_breakdown": True,
        "train_micro_batch_size_per_gpu": 8,
        "parquet_config": {
            "token": 1
        }
    }

    engine, _, _, _ = DeepSpeedEngine.initialize(args=ds_config, model=model, optimizer=None)

if __name__ == "__main__":
    main()