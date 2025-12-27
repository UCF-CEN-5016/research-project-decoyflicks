import labml
from labml import monit, tracker, experiment
from labml_nn.optimizers.Adam import Adam
from labml_nn.optimizers.WeightDecay import WeightDecay
import torch

# Define x parameter
x = torch.Parameter(torch.tensor([.0]))
# Optimal, x_star = -1
x_star = torch.tensor([-1], requires_grad=False)

def func(t: int, x_: torch.Parameter):
    if t % 101 == 1:
        return (1010 * x_).sum()
    else:
        return (-10 * x_).sum()

# Initialize the relevant optimizer
optimizer = Adam([x], lr=1e-2, betas=(0.9, 0.99))
# R(T)
total_regret = 0

with experiment.record(name='synthetic', comment='Adam'):
    for step in monit.loop(10_000_000):
        regret = func(step, x) - func(step, x_star)
        total_regret += regret.item()
        if (step + 1) % 1000 == 0:
            tracker.save(loss=regret, x=x, regret=total_regret / (step + 1))
        regret.backward()
        optimizer.step()
        optimizer.zero_grad()
        x.data.clamp_(-1., +1.)

# Change the optimizer to AMSGrad and repeat the steps above
optimizer = Adam([x], lr=1e-2, betas=(0.9, 0.99))
total_regret = 0

with experiment.record(name='synthetic', comment='AMSGrad'):
    for step in monit.loop(10_000_000):
        regret = func(step, x) - func(step, x_star)
        total_regret += regret.item()
        if (step + 1) % 1000 == 0:
            tracker.save(loss=regret, x=x, regret=total_regret / (step + 1))
        regret.backward()
        optimizer.step()
        optimizer.zero_grad()
        x.data.clamp_(-1., +1.)