import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification

def setup_distributed_training():
    # Initialize the process group for DDP
    torch.distributed.init_process_group(backend='nccl')

    # Load a model (replace with your actual model)
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    
    # Wrap the model with DDP
    ddp_model = DDP(model)

    return ddp_model

def train_model(ddp_model):
    # Set the model to training mode
    ddp_model.train()

    # Create a dummy input and target
    inputs = torch.randint(0, 256, (8, 128))
    targets = torch.randint(0, 2, (8,))

    # Forward pass
    outputs = ddp_model(inputs)

    # Compute loss
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Check if the parameter is updating
    print(ddp_model.module.classifier.weight.grad is not None)

if __name__ == "__main__":
    model = setup_distributed_training()
    train_model(model)