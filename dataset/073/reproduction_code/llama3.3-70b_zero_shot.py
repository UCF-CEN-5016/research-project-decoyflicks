import deepspeed
import torch
import torch.distributed as dist

class CDB:
    def __init__(self):
        pass

    def all_reduce(self, tensor, op, group, async_op):
        pass

def main():
    cdb = None
    tensor = torch.randn(10)
    op = dist.ReduceOp.SUM
    group = None
    async_op = False
    cdb.all_reduce(tensor, op, group, async_op)

if __name__ == "__main__":
    main()