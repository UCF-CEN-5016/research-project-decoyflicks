# Example code showing how to fix the num_classes issue for CIFAR-10

import torch
from vision_transformer import create_model, train, test

def main():
    # Change the number of classes to match the dataset (CIFAR-10 has 10 classes)
    model = create_model(num_classes=10)
    
    # You can continue with training and testing as before
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model, train_acc, val_acc = train(model, device=device)
    test(model, device=device)

if __name__ == '__main__':
    main()