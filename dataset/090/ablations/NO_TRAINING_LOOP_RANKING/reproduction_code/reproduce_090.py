import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from your_module import MultiInputTransformerWrapper, TransformerWrapper, Encoder, Decoder

def main():
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    input_shape = (2, 1024)

    x = {
        'note': torch.randint(0, 20000, input_shape).to(device),
        'pitch': torch.randint(0, 32, input_shape).to(device),
        'tone': torch.randint(0, 16, input_shape).to(device)
    }

    model = MultiInputTransformerWrapper(
        num_tokens={'note': 20000, 'pitch': 32, 'tone': 16},
        max_seq_len=1024,
        return_only_embed=True,
        attn_layers=Decoder(dim=128, depth=6, heads=8)
    ).to(device)

    embed = model(x)
    assert embed.shape == (2, 1024, 128)

    model = TransformerWrapper(
        num_tokens=20000,
        max_seq_len=1024,
        num_memory_tokens=2,
        average_pool_embed=True,
        attn_layers=Encoder(dim=128, depth=6, heads=8)
    ).to(device)

    x_input = torch.randint(0, 20000, (2, 1024)).to(device)
    mask = torch.randint(0, 2, (2, 1024)).bool().to(device)

    logits = model(x_input, mask=mask)
    assert logits.shape == (2, 20000)

    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name)

    if 'encoder.to_logits.weight' in model.state_dict():
        print("encoder.to_logits.weight is present in the model parameters.")

    model = DDP(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(x_input, mask=mask)
        loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 20000, (2,)).to(device))
        loss.backward()
        optimizer.step()

        print("Gradients of encoder.to_logits.weight:")
        print(model.module.encoder.to_logits.weight.grad)

if __name__ == "__main__":
    main()