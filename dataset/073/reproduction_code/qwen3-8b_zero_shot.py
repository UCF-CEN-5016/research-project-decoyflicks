import torch
import deepspeed
import torch.distributed as dist

def main():
    dist.init_process_group(backend='nccl')
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "enabled": True
        }
    }
    model, optimizer, _ = deepspeed.initialize(model, optimizer, config)
    tensor = torch.randn(1)
    cdb = model.deepspeed
    cdb.all_reduce(tensor)

if __name__ == "__main__":
    main()