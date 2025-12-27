import torch
import random

def main():
    x = torch.randn(2, 2, requires_grad=True)
    t = torch.tensor([0.5])

    for _ in range(10):
        if random() < 0.5:
            with torch.inference_mode():
                # Clone and detach x to ensure it's not part of the computation graph
                x_self_cond = x.clone().detach()
                # Use x_self_cond in a way that requires gradients
                loss = (x_self_cond + x).square().mean()
                # This line will raise an error because x_self_cond is detached
                loss.backward()

if __name__ == "__main__":
    main()

import torch
import random

def main():
    x = torch.randn(2, 2, requires_grad=True)
    t = torch.tensor([0.5])

    for _ in range(10):
        if random() < 0.5:
            with torch.no_grad():
                # Clone and detach x
                x_self_cond = x.clone().detach()
                # Use x_self_cond in a way that requires gradients
                loss = (x_self_cond + x).square().mean()
                # This will not raise an error
                loss.backward()

if __name__ == "__main__":
    main()