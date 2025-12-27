import os
import pytorch_lightning as pl
import nnunet
import torch
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

learning_rate = 0.0003
epochs = 10
fold = 0
gpus = 1
task = 11
nfolds = 10
batch_size = 32

train_data = torch.randn(1125, 3, 128, 128)
val_data = torch.randn(126, 3, 128, 128)
test_data = torch.randn(1251, 3, 128, 128)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = nnunet.NNUNet()

trainer = pl.Trainer(max_epochs=epochs, gpus=gpus)

try:
    trainer.fit(model, train_loader, val_loader)
except Exception as e:
    if 'MisconfigurationException' in str(e):
        print(str(e))