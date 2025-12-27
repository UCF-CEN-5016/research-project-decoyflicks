import torch

class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, ...):
        # implementation of the wrapper model

net = ContinuousTransformerWrapper(
    # initialization parameters...
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]
logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([m.shape for m in mems])