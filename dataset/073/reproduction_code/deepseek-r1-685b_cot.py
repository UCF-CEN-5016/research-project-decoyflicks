import torch
import torch.distributed as dist

# Minimal reproduction of the Domino/DeepSpeed cdb initialization bug
def reproduce_bug():
    # Initialize distributed environment (mimicking real setup)
    dist.init_process_group(backend='nccl')
    
    # Simulate uninitialized cdb - this would normally be set by DeepSpeed/Domino
    cdb = None
    
    # Create a test tensor (like in actual training)
    tensor = torch.ones(2, 2).cuda()
    
    try:
        # Attempt all_reduce operation - will fail with AttributeError
        result = cdb.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print("Operation succeeded (unexpected)")
    except AttributeError as e:
        print(f"Reproduced bug successfully! Error: {e}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    reproduce_bug()