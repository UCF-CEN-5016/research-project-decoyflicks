import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from your_module import MultiInputTransformerWrapper, Decoder  # Replace 'your_module' with the actual module name

def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = MultiInputTransformerWrapper(
        num_tokens=dict(
            note=20000,
            pitch=32,
            tone=16
        ),
        max_seq_len=1024,
        return_only_embed=True,
        attn_layers=Decoder(
            dim=128,
            depth=6,
            heads=8
        )
    ).to(rank)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    x = dict(
        note=torch.randint(0, 20000, (2, 1024)).to(rank),
        pitch=torch.randint(0, 32, (2, 1024)).to(rank),
        tone=torch.randint(0, 16, (2, 1024)).to(rank)
    )
    
    print("Before forward pass:", model.module.attn_layers.to_logits.weight.data)
    embed = model(x)
    print("After forward pass:", model.module.attn_layers.to_logits.weight.data)
    
    assert model.module.attn_layers.to_logits.weight.data.equal(model.module.attn_layers.to_logits.weight.data)

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)