import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

def train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        for inputs, labels in val_loader:
            with torch.no_grad():
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), labels.float()).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss - {loss.item():.4f}, Val Loss - {avg_val_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Generate synthetic data with clear class separation
    X, y = make_circles(n_samples=1000, noise=0.1)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create PyTorch tensors
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), 
                                                torch.from_numpy(y_train).float())
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                               torch.from_numpy(1 - y_val).float())  # Invert labels for dog class
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)  # Don't shuffle during validation
    
    # Initialize model and optimizer
    model = create_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Run training loop
    train_and_validate(model, optimizer, criterion, train_loader, val_loader)