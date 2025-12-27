import torch
import random

def main():
    x = torch.randn(2, 2, requires_grad=True)

    for _ in range(10):
        if random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = x.detach().clone()
                loss = (x_self_cond + x).pow(2).mean()
                loss.backward()

if __name__ == "__main__":
    main()