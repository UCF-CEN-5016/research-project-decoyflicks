import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

class MyTrainDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(20), torch.randint(0, 2, (1,))

def load_train_objs():
    train_set = MyTrainDataset(100)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

def main():
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, 32)
    print(torch.backends.mps.is_available())  # This line will trigger the bug

if __name__ == "__main__":
    main()