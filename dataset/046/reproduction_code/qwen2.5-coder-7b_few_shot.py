import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_model(in_features: int = 10, out_features: int = 50) -> nn.Module:
    return LinearClassifier(in_features, out_features)


def run_sample_inference(model: nn.Module, batch_size: int = 32, feature_dim: int = 10) -> torch.Tensor:
    batch = torch.randn(batch_size, feature_dim)
    outputs = model(batch)
    return outputs


def validate_output_dim(tensor: torch.Tensor, expected_dim: int = 3) -> None:
    if tensor.dim() != expected_dim:
        raise RuntimeError("log_probs must be 3-D (batch_size, input length, num classes)")


def main() -> None:
    classifier = create_model(in_features=10, out_features=50)
    outputs = run_sample_inference(classifier, batch_size=32, feature_dim=10)
    validate_output_dim(outputs)


if __name__ == "__main__":
    main()