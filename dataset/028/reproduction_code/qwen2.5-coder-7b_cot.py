import torch
from vision_transformer import create_model, train, test

def choose_device() -> str:
    """Return 'cuda' if a CUDA device is available, otherwise 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_model(num_classes: int = 10):
    """Create and return a Vision Transformer model configured for the given number of classes."""
    return create_model(num_classes=num_classes)

def run_training(model, device: str):
    """Run training on the model and return the trained model and accuracy metrics."""
    trained_model, train_accuracy, val_accuracy = train(model, device=device)
    return trained_model, train_accuracy, val_accuracy

def run_evaluation(model, device: str):
    """Run testing/evaluation on the trained model."""
    test(model, device=device)

def main():
    vit = prepare_model(num_classes=10)  # CIFAR-10 has 10 classes
    compute_device = choose_device()
    vit, train_accuracy, val_accuracy = run_training(vit, device=compute_device)
    run_evaluation(vit, device=compute_device)

if __name__ == '__main__':
    main()