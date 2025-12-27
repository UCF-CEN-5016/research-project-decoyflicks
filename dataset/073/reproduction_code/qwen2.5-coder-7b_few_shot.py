import torch
import deepspeed

def create_random_tensor(length: int = 10) -> torch.Tensor:
    """Create a 1-D tensor with random values drawn from a normal distribution."""
    return torch.randn(length)

def init_cpu_offload_backend():
    """Initialize and return the DeepSpeed CPU offload backend."""
    return deepspeed.CPUoffload()

def run_training():
    """Run the (simulated) distributed training step that performs an all-reduce."""
    tensor = create_random_tensor(10)
    cpu_offload_backend = init_cpu_offload_backend()
    cpu_offload_backend.all_reduce(tensor)

if __name__ == "__main__":
    run_training()