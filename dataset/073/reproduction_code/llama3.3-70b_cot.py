import torch
import torch.distributed as dist

# Minimal setup: create a distributed process group
def init_process_group(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

# Simulate the cdb object not being initialized
cdb = None

# Triggering conditions: attempt to use the all_reduce method on the cdb object
def all_reduce(tensor):
    # This will raise an AttributeError because cdb is None
    return cdb.all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False)

# Example usage
if __name__ == "__main__":
    rank = 0
    world_size = 1
    init_process_group(rank, world_size)

    tensor = torch.tensor([1.0])
    try:
        all_reduce(tensor)
    except AttributeError as e:
        print(f"[rank{rank}]: {e}")