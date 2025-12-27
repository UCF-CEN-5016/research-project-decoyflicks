import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Convolutional discriminator that reduces a 128x128 RGB image down to a single
    sigmoid-activated scalar per example (shape: [B, 1, 1, 1]).
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = self._make_feature_extractor()
        self.classifier_head = self._make_classifier_head()

    def _make_feature_extractor(self) -> nn.Sequential:
        # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def _make_classifier_head(self) -> nn.Sequential:
        # 8x8 -> 4x4 -> AdaptiveAvgPool -> 1x1, then sigmoid
        return nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        output = self.classifier_head(features)
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    inputs = torch.rand(batch_size, 3, 128, 128).to(device)

    model = Discriminator().to(device)
    outputs = model(inputs)

    targets = torch.rand(batch_size, 1, 1, 1).to(device)
    loss = F.binary_cross_entropy(outputs, targets)

    print(loss)