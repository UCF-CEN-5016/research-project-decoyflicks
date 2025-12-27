import os
import subprocess
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from nvidia import BertConfig, BertForSequenceClassification
import deepspeed

def main():
    os.system("git clone https://github.com/microsoft/DeepSpeedExamples.git")
    os.chdir("DeepSpeedExamples/training/DeepSpeed-Domino")
    os.system("conda create -n deepspeed_env python=3.12 -y")
    os.system("conda activate deepspeed_env")
    os.system("module load openmpi/4.1.1")
    os.system("module load cuda/11.8.0")
    os.system("module load gcc/10.2.0")

    job_script = """#!/bin/bash
#SBATCH --job-name="deepspeed_example"
#SBATCH --time=24:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[41-42]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240G
#SBATCH --output=deepspeed_example.%j.o
#SBATCH --error=deepspeed_example.%j.e
#SBATCH --account=g34

srun nvidia-smi -L
srun bash pretrain_gpt3_2.7b.sh
"""
    with open("deepspeed_example.sh", "w") as f:
        f.write(job_script)

    pretrain_script = """#!/bin/bash
cd /users/zhu/DeepSpeedExamples/training/DeepSpeed-Domino
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=1
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
CHECKPOINT_PATH=checkpoint
rm -rf $CHECKPOINT_PATH/*
VOCAB_FILE="dataset/gpt2-vocab.json"
MERGE_FILE="dataset/gpt2-merges.txt"
DATA_PATH="dataset/my-gpt2_text_document"

GPT_ARGS="--num-layers 12 --hidden-size 2560 --num-attention-heads 32 --seq-length 512 --max-position-embeddings 512 --micro-batch-size 32 --global-batch-size 64 --lr 0.00015 --train-iters 10 --lr-decay-iters 320000 --lr-decay-style cosine --min-lr 1.0e-5 --weight-decay 1e-2 --lr-warmup-fraction .01 --clip-grad 1.0 --fp16 --tensor-model-parallel-size $GPUS_PER_NODE --distributed-backend nccl --seed 3407"
DATA_ARGS="--data-path $DATA_PATH --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --split 949,50,1"
OUTPUT_ARGS="--log-interval 1"

deepspeed --num_gpus=$GPUS_PER_NODE pretrain_gpt3_2.7b.py $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS
"""
    with open("pretrain_gpt3_2.7b.sh", "w") as f:
        f.write(pretrain_script)

    os.system("sbatch deepspeed_example.sh")

if __name__ == "__main__":
    main()