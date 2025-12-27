import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.runtime.domino.transformer import TransformerModuleParallel


MODEL_ID = "gpt2"
_DEEPSPEED_CONFIG = {
    "zero_optimizers": {"parameter_sharding": True},
    "fp16": True,
}


def load_tokenizer_and_model(model_id: str):
    """
    Load tokenizer and causal LM model from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model


def get_compute_device() -> torch.device:
    """
    Return the appropriate compute device (GPU if available, otherwise CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_module_parallel(model, deepspeed_config: dict):
    """
    Wrap the model with DeepSpeed's TransformerModuleParallel using the provided config.
    """
    return TransformerModuleParallel(model, deepspeed_config)


def prepare_random_inputs(batch_size: int = 1, seq_len: int = 512):
    """
    Prepare random input tensor and attention mask matching the expected shape.
    """
    inputs = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    return inputs, attention_mask


def run_inference_parallel(module_parallel, inputs: torch.Tensor, attention_mask: torch.Tensor):
    """
    Run the given module in parallel with the provided inputs and attention mask.
    """
    return module_parallel(inputs, attention_mask=attention_mask)


def main():
    tokenizer, model = load_tokenizer_and_model(MODEL_ID)

    device = get_compute_device()
    model.to(device)

    module_parallel = initialize_module_parallel(model, _DEEPSPEED_CONFIG)

    inputs, attention_mask = prepare_random_inputs(batch_size=1, seq_len=512)
    outputs = run_inference_parallel(module_parallel, inputs, attention_mask=attention_mask)

    return outputs


if __name__ == "__main__":
    main()