import os
import torch
import pytorch_lightning as pl
from nnunet import nnUNet

# Set up training parameters
learning_rate = 0.0003
epochs = 10
fold = 0
gpus = 1
task = 11
nfolds = 10

# Prepare the dataset
# Ensure the BraTS dataset is correctly formatted and accessible

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(gpus=gpus)

# Start training
try:
    trainer.fit(nnUNet(task=task, nfolds=nfolds, fold=fold, learning_rate=learning_rate, epochs=epochs))
except Exception as e:
    print(e)