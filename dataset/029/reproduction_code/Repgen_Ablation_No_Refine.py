import argparse
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module

def demo_tp(rank, args):
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)
    device_mesh = DeviceMesh("cuda", torch.arange(args.world_size))
    model = ToyModel().cuda(rank)
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    model = parallelize_module(model, device_mesh, PairwiseParallel())
    
    for _ in range(args.iter_nums):
        inp = torch.rand(20, 10).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()

def run_demo(demo_fn, args):
    from multiprocessing import Process
    processes = []
    for rank in range(args.world_size):
        p = Process(target=demo_fn, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 32)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(32, 5)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    args = parser.parse_args()
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    elif not hasattr(torch.backends, 'mps'):
        print(
            "PyTorch doesn't have MPS available, need nightly build."
        )
    else:
        run_demo(demo_tp, args)