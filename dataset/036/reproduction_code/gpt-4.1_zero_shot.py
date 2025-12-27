import torch

rewards = torch.tensor([1., 2., 3., 4.])
values = torch.tensor([0.5, 1.0, 1.5, 2.0, 0.0])
gamma = 0.9
_lam = 0.95

# Original bug: uses gamma^(k) * r_{t+1} instead of r_{t+2}
returns_bug = []
for t in range(len(rewards)):
    ret = 0
    for k in range(1, 3):
        ret += (gamma**k) * rewards[min(t+1, len(rewards)-1)]
    returns_bug.append(ret)
returns_bug = torch.tensor(returns_bug)

# Correct calculation: gamma^2 * r_{t+2}
returns_correct = []
for t in range(len(rewards)):
    ret = 0
    for k in range(1, 3):
        idx = min(t + k, len(rewards) - 1)
        ret += (gamma**k) * rewards[idx]
    returns_correct.append(ret)
returns_correct = torch.tensor(returns_correct)

# Original bug: weights wk = lambda^k instead of (1-lambda)*lambda^(k-1)
weights_bug = [( _lam**k ) for k in range(1, 5)]

# Correct weights: (1-lambda)*lambda^(k-1)
weights_correct = [(1 - _lam) * (_lam**(k - 1)) for k in range(1, 5)]

print("Returns bug:", returns_bug)
print("Returns correct:", returns_correct)
print("Weights bug:", weights_bug)
print("Weights correct:", weights_correct)