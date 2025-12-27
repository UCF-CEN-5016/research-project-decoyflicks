import os
import torch

def main():
    print("Initializing MaskRCNN model")
    
    # Initialize model with random weights (dummy training)
    model = modellib.MaskRCNN(mode="inference", model_dir=None, config=config)
    
    print("Training the model for 10 epochs...")
    optimizer = torch.optim.Adam(model.parameters())
    dummy_data = torch.randn(32, 10)  # Sample input
    dummy_labels = torch.randint(0, 2, (32,))  # Simple labels
    
    for epoch in range(10):
        with torch.no_grad():
            outputs = model(dummy_data)
            loss = torch.nn.functional.mse_loss(outputs, dummy_labels.unsqueeze(-1))
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    print("Saving model using PyTorch format")
    torch.save(model.state_dict(), os.path.join("model_weights", "maskrcnn.pth"))

    print("Loading weights into new instance of the same model")
    # This should raise the NotImplementError mentioned in your issue
    loaded_model = modellib.MaskRCNN(mode="inference", model_dir=None, config=config)
    try:
        loaded_model.load_weights("model_weights/maskrcnn.pth", by_name=True)
        print("Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"\nRunning on {device}")
    main()