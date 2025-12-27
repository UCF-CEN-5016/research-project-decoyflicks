import torch
import torch.distributed as dist
import deepspeed.comm as cdb

def init_distributed():
    dist.init_process_group(backend='nccl')
    cdb.initialize()

def test_all_reduce():
    tensor = torch.ones(1).cuda()
    cdb.all_reduce(tensor)

if __name__ == "__main__":
    init_distributed()
    test_all_reduce()