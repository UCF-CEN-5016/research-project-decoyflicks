import os
import sys
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

command = f'python ../main.py --brats --brats22_model --scheduler --learning_rate {learning_rate} --epochs {epochs} --fold {fold} --gpus {gpus} --task {task} --nfolds {nfolds} --save_ckpt'

os.system('pip install -r requirements.txt')
assert torch.__version__ == '2.2.0+cu121'
assert torch.cuda.is_available() == True

output = os.popen(command).read()
assert 'MisconfigurationException' in output
assert 'CosineAnnealingWarmRestarts' in output
print(output)
assert 'WSL2' in sys.platform
assert torch.cuda.memory_allocated() > 0