import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.runtime.domino.transformer import TransformerModuleParallel

def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize DeepSpeed
    deepspeed_config = {
        "zero_optimizers": {"parameter_sharding": True},
        "fp16": True,
    }
    transformer_module_parallel = TransformerModuleParallel(model, deepspeed_config)

    # Run the model in parallel
    inputs = torch.randn(1, 512)
    attention_mask = torch.ones((1, 512))
    outputs = transformer_module_parallel(inputs, attention_mask=attention_mask)

if __name__ == "__main__":
    main()