import torch
from torch.nn.functional import log_softmax
from torch.utils.data.dataloader import DataLoader

torch.manual_seed(42)


def build_dataset():
    # Original sample data structure preserved
    return [[["a", "b"], [0, 1]]]


def build_log_probabilities():
    # Preserve the original tensor values
    return torch.tensor([[[0.5, 0.5], [0.3, 0.7]]])


def process_dataloader(dataset, batch_size: int = 2):
    loader = DataLoader(dataset, batch_size=batch_size)
    for batch in loader:
        # Maintain the same operation on the second element of the batch
        log_softmax(batch[1], dim=-1)


def process_tensor_probs(probs_tensor: torch.Tensor):
    # Maintain the same operation on the standalone tensor
    log_softmax(probs_tensor, dim=-1)


def main():
    sample_data = build_dataset()
    probs = build_log_probabilities()

    process_dataloader(sample_data, batch_size=2)
    process_tensor_probs(probs)


if __name__ == "__main__":
    main()