import torch
import torch.distributed as dist

def faulty_all_reduce():
    # Missing dist.init_process_group() causes dist._default_pg to be None
    tensor = torch.ones(1)
    # Attempt to call all_reduce on uninitialized backend
    dist._default_pg.all_reduce(tensor)  # dist._default_pg is None

if __name__ == "__main__":
    faulty_all_reduce()