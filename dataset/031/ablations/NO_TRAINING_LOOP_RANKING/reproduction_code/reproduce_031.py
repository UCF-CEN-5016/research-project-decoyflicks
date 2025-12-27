import os
import paddle
import logging
from modeling import BertForPretraining, BertConfig
from utils.save_load import save_model
from utils.utility import get_trainer_id
from lr_scheduler import build_lr_scheduler
from optimizer import build_optimizer

def main():
    os.system("git clone https://github.com/NVIDIA/DeepLearningExamples.git")
    os.chdir("DeepLearningExamples/PyTorch/Classification/ConvNets/resnet50v1.5")
    os.system("pip install -r requirements.txt")
    
    pretrained_weights_path = "nvidia_resnet50_200821.pth.tar"
    os.system(f"wget <URL_TO_PRETRAINED_WEIGHTS> -O {pretrained_weights_path}")
    
    dataset_path = "/data/imagenet/"
    batch_size = 256
    command = f"python ./main.py --arch=resnet50 --evaluate --epochs=1 --pretrained-from-file={pretrained_weights_path} -b={batch_size} {dataset_path}"
    
    os.system(command)

if __name__ == "__main__":
    main()