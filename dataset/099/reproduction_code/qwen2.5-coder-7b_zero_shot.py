import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model_and_tokenizer(pretrained_name: str = "t5-small"):
    """
    Load a T5 model and its tokenizer from the HuggingFace hub.
    """
    tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_name)
    return model, tokenizer


def build_example_inputs():
    """
    Construct example input_ids and attention_mask tensors.
    """
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    return input_ids, attention_mask


def run_model_forward(model: T5ForConditionalGeneration, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Run the model forward pass with the provided inputs.
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    return outputs


def compute_loss_from_logits(logits: torch.Tensor, target: torch.Tensor):
    """
    Compute cross-entropy loss between model logits and a target tensor.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)
    return loss


def main():
    model, tokenizer = load_model_and_tokenizer("t5-small")
    input_ids, attention_mask = build_example_inputs()
    outputs = run_model_forward(model, input_ids, attention_mask)

    target = torch.tensor([[2]])
    loss = compute_loss_from_logits(outputs.logits, target)

    print(loss)


if __name__ == "__main__":
    main()