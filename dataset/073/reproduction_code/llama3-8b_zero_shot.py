import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Model(DDP):
    pass

model = Model(model=None)
dist.init_process_group(backend='nccl')