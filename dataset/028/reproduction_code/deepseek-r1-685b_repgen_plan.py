import os
import threading
import time
from datetime import datetime

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.distributed.rpc import RRef, rpc_async, rpc_sync

import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import torch.multiprocessing as mp


class TrainingConfig:
    """
    Configuration class for distributed training parameters.
    """
    def __init__(self,
                 batch_size: int = 20,
                 image_w: int = 64,
                 image_h: int = 64,
                 num_classes: int = 10, # CIFAR10 has 10 classes
                 batch_update_size: int = 5,
                 master_addr: str = 'localhost',
                 master_port: str = '29500',
                 rpc_timeout_seconds: int = 0, # 0 means infinite timeout
                 num_batches_per_epoch: int = 2): # Example: limit training to a few batches per epoch for demonstration
        self.batch_size = batch_size
        self.image_w = image_w
        self.image_h = image_h
        self.num_classes = num_classes
        self.batch_update_size = batch_update_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.num_batches_per_epoch = num_batches_per_epoch

def timed_log(text: str):
    """Prints a log message with a timestamp."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")

class BatchUpdateParameterServer(object):
    """
    Parameter Server (PS) responsible for holding the global model,
    accumulating gradients from trainers, and updating the model in batches.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = torchvision.models.resnet50(num_classes=self.config.num_classes)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self) -> nn.Module:
        """Returns the current state of the global model."""
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref: RRef, grads: list[torch.Tensor]) -> torch.futures.Future:
        """
        Receives gradients from a trainer, accumulates them, and updates the model
        when enough gradients are received. Returns a Future that resolves with
        the updated model.
        Args:
            ps_rref (RRef): RRef to the Parameter Server instance.
            grads (list[torch.Tensor]): List of gradients from a trainer.
        """
        self = ps_rref.local_value()
        timed_log(f"PS got {self.curr_update_size + 1}/{self.config.batch_update_size} updates")

        for p, g in zip(self.model.parameters(), grads):
            p.grad += g.cuda() if torch.cuda.is_available() else g

        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.config.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.config.batch_update_size
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)
                
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
                
                fut.set_result(self.model)
                timed_log("PS updated model and sent to trainers")

        return fut

class Trainer(object):
    """
    Trainer class responsible for fetching the model from PS, training on a
    local batch, and sending gradients back to the PS.
    """
    def __init__(self, ps_rref: RRef, config: TrainingConfig):
        self.ps_rref = ps_rref
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        
        transform = transforms.Compose([
            transforms.Resize((self.config.image_w, self.config.image_h)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2 or 1
        )
        self.data_iterator = iter(self.dataloader)

    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches the next batch of data from the DataLoader."""
        try:
            inputs, labels = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            inputs, labels = next(self.data_iterator)
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        return inputs, labels

    def train(self):
        """
        Main training loop for a single trainer worker.
        Fetches model, computes loss, backpropagates, and sends gradients to PS.
        """
        worker_name = rpc.get_worker_info().name
        
        current_model = rpc_sync(self.ps_rref.owner(), BatchUpdateParameterServer.get_model, args=(self.ps_rref,))
        if torch.cuda.is_available():
            current_model = current_model.cuda()
        
        for batch_idx in range(self.config.num_batches_per_epoch):
            inputs, labels = self.get_next_batch()
            timed_log(f"{worker_name} processing batch {batch_idx + 1}/{self.config.num_batches_per_epoch}")
            
            current_model.zero_grad()
            
            outputs = current_model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            
            timed_log(f"{worker_name} reporting grads for batch {batch_idx + 1}")
            
            grads_to_send = [p.grad.cpu() for p in current_model.parameters()]
            
            current_model = rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, grads_to_send),
            )
            if torch.cuda.is_available():
                current_model = current_model.cuda()
            timed_log(f"{worker_name} got updated model for batch {batch_idx + 1}")

def run_trainer_process(ps_rref: RRef, config: TrainingConfig):
    """Entry point for trainer processes."""
    trainer = Trainer(ps_rref, config)
    trainer.train()

def run_ps_process(trainer_names: list[str], config: TrainingConfig):
    """Entry point for the parameter server process."""
    timed_log("Parameter Server: Starting training coordination")
    ps_rref = RRef(BatchUpdateParameterServer(config))
    
    futs = []
    for trainer_name in trainer_names:
        futs.append(
            rpc_async(trainer_name, run_trainer_process, args=(ps_rref, config))
        )

    torch.futures.wait_all(futs)
    timed_log("Parameter Server: All trainers finished. Training complete.")

def run_distributed_training(rank: int, world_size: int, config: TrainingConfig):
    """
    Initializes the RPC framework and starts either the PS or a trainer process.
    """
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    
    rpc_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=config.rpc_timeout_seconds
    )

    if rank == 0:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_options
        )
        trainer_names = [f"trainer{r}" for r in range(1, world_size)]
        run_ps_process(trainer_names, config)
    else:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_options
        )

    rpc.shutdown()

if __name__ == "__main__":
    training_config = TrainingConfig(
        batch_size=20,
        image_w=64,
        image_h=64,
        num_classes=10,
        batch_update_size=5,
        master_addr='localhost',
        master_port='29500',
        rpc_timeout_seconds=0,
        num_batches_per_epoch=2
    )
    
    world_size = training_config.batch_update_size + 1
    
    print(f"Starting distributed training with world_size={world_size} (1 PS + {training_config.batch_update_size} trainers)")
    
    mp.spawn(run_distributed_training, args=(world_size, training_config), nprocs=world_size, join=True)
