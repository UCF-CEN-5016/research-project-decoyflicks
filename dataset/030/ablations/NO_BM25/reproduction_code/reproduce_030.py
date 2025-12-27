import os
import pytorch_lightning as pl
import nnunet
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

learning_rate = 0.0003
epochs = 10
fold = 0
gpus = 1
task = 11
nfolds = 10

# Mock dataset creation
train_data = torch.randn(1125, 1, 128, 128, 128)  # Example shape for BraTS
train_labels = torch.randint(0, 2, (1125, 1, 128, 128, 128))

val_data = torch.randn(126, 1, 128, 128, 128)
val_labels = torch.randint(0, 2, (126, 1, 128, 128, 128))

test_data = torch.randn(1251, 1, 128, 128, 128)

# Configuration for nnUNet
config = {
    'scheduler': True,
    'learning_rate': learning_rate,
    'epochs': epochs,
    'fold': fold,
    'gpus': gpus,
    'task': task,
    'nfolds': nfolds
}

# Save mock dataset to disk (not shown in the original context, but necessary for nnUNet)
# Assuming nnUNet has a method to save datasets in the expected format

# Run training script
os.system(f'python ../main.py --brats --brats22_model --scheduler --learning_rate {learning_rate} --epochs {epochs} --fold {fold} --gpus {gpus} --task {task} --nfolds {nfolds} --save_ckpt')