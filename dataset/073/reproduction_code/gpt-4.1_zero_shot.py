import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')
tensor = torch.tensor([1.0]).cuda()
cdb = None
cdb.all_reduce(tensor)