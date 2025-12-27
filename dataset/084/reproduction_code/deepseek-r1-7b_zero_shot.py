import torch
from torch.utils.data import DataLoader

class ToyDataset(Dataset):
    def __init__(self, num_samples=313):
        self.num_samples = num_samples
        # Generate toy data where each sample's input directly corresponds to its label
        # Each input is a vector that will be mapped to the corresponding class (0 or 1)
        self.x = torch.zeros(num_samples, 2)  # Assuming binary classification with two features
        self.y = torch.randint(0, 2, (num_samples,))  # Random binary labels
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
    import time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 5)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set to training mode
            else:
                model.eval()   # Set to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                # Backward pass and optimize only if training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # deep copy model
        if best_acc < epoch_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Top validation accuracy: {best_acc}')


# Example usage with the ToyDataset
def main():
    # Initialize model, criterion, and optimizer
    model = ToyModel()  # Replace ToyModel with your actual model class
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create toy dataset
    train_dataset = ToyDataset(num_samples=313)  # Adjust number as needed
    
    # Batch size can be adjusted; for this example, use full batch
    batch_size = 313
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10
    
    train_model(model, criterion, optimizer, {'train': train_loader, 'val': None}, num_epochs=num_epochs)

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader

class ToyDataset(Dataset):
    def __init__(self, num_samples=313):
        self.num_samples = num_samples
        # Generate toy data where each sample's input directly corresponds to its label
        # Each input is a vector that will be mapped to the corresponding class (0 or 1)
        self.x = torch.zeros(num_samples, 2)  # Assuming binary classification with two features
        self.y = torch.randint(0, 2, (num_samples,))  # Random binary labels
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
    import time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 5)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set to training mode
            else:
                model.eval()   # Set to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                # Backward pass and optimize only if training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # deep copy model
        if best_acc < epoch_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Top validation accuracy: {best_acc}')


# Example usage with the ToyDataset
def main():
    import torch.nn as nn
    from toy_model import ToyModel  # Replace with your actual model
    
    # Initialize model, criterion, and optimizer
    model = ToyModel() 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create toy dataset
    train_dataset = ToyDataset(num_samples=313)  # Adjust number as needed
    
    # Batch size can be adjusted; for this example, use full batch
    batch_size = 313
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 10
    
    train_model(model, criterion, optimizer, {'train': train_loader, 'val': None}, num_epochs=num_epochs)

if __name__ == "__main__":
    main()