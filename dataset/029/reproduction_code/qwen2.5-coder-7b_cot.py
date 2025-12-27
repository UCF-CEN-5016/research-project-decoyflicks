import argparse
import sys
from typing import Optional

import torch
import torchvision.models as tv_models


def _mps_backend_available() -> bool:
    """
    Safely check whether the MPS backend is available on this torch installation.
    This avoids AttributeError on torch.backends when 'mps' does not exist.
    """
    mps = getattr(torch.backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def choose_device(prefer_cuda: bool = False) -> torch.device:
    """
    Choose the device to run on:
      - CUDA if prefer_cuda is True and CUDA is available
      - MPS if available
      - otherwise CPU

    This mirrors the typical selection logic used in many PyTorch examples,
    but handles torch installations that do not expose torch.backends.mps.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_backend_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str = "resnet18", num_classes: Optional[int] = None) -> torch.nn.Module:
    """
    Construct a model by name. Currently supports 'resnet18' and falls back with a clear error.
    """
    name = model_name.lower()
    if name == "resnet18":
        # Keep using the familiar pretrained=False parameter for broad compatibility.
        model = tv_models.resnet18(pretrained=False)
        if num_classes is not None:
            # Replace the final layer if a different number of classes is requested.
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def parse_arguments(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored example: select device and build a ResNet18 model.")
    parser.add_argument("--model", default="resnet18", help="Model name (default: resnet18)")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of output classes (optional)")
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA if available")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_arguments(argv or sys.argv[1:])
    device = choose_device(prefer_cuda=args.prefer_cuda)

    model = build_model(args.model, num_classes=args.num_classes)
    model.to(device)

    # Example feedback for the user; keeps core behavior intact while being explicit.
    print(f"Using device: {device}")
    print(f"Built model: {args.model} (num_classes={args.num_classes})")


if __name__ == "__main__":
    main()