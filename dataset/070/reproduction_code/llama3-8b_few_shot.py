import torch
from deepspeed.runtime.domino.transformer import TransformerModelParallel, DominoBatching
from deepspeed.comm import comm


def main():
    # Initialize a DeepSpeed model
    model = TransformerModelParallel(config=..., input_size=512, hidden_size=2560)

    # Set the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a DominoBatching object
    batcher = DominoBatching(model=model, max_sequence_length=512)

    # Create a tensor to simulate input data
    input_tensor = torch.randn(1, 512, 2560).to(device)

    # Simulate the forward pass
    output = model(input_tensor, attention_mask=torch.ones_like(input_tensor))

    # Call all_reduce on the output tensor
    comm.all_reduce(output)

if __name__ == "__main__":
    main()

import deepspeed

def main():
    # Initialize a DeepSpeed model
    model = TransformerModelParallel(config=..., input_size=512, hidden_size=2560)

    # Set the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a DominoBatching object
    batcher = DominoBatching(model=model, max_sequence_length=512)

    # Create a tensor to simulate input data
    input_tensor = torch.randn(1, 512, 2560).to(device)

    # Initialize the DeepSpeed runtime
    deepspeed.runtime.init()

    # Simulate the forward pass
    output = model(input_tensor, attention_mask=torch.ones_like(input_tensor))

    # Call all_reduce on the output tensor
    comm.all_reduce(output)

if __name__ == "__main__":
    main()