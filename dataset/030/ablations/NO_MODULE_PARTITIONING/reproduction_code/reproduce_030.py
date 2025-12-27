import torch
import pytorch_lightning as pl
from nnunet import nnUNet
from torch.utils.data import DataLoader
from apex import amp
import hydra

def main():
    torch.manual_seed(42)
    
    learning_rate = 0.0003
    epochs = 10
    fold = 0
    gpus = 1
    task_id = 11

    train_dataset, valid_dataset = hydra.utils.call('dataset')
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, num_workers=4)

    model = nnUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)
    
    try:
        trainer.fit(model, train_dataloader, valid_dataloader)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()