import torch

def compute_gae_buggy(rewards, values, gamma, lmbda):
    """
    Compute GAE with buggy formula:
    - uses gamma^2 * r_{t+1} instead of gamma^2 * r_{t+2}
    - uses w_k = lambda^k instead of w_k = (1-lambda) * lambda^{k-1}
    Args:
        rewards (torch.Tensor): shape (T,)
        values (torch.Tensor): shape (T+1,)
        gamma (float): discount factor
        lmbda (float): GAE lambda
    Returns:
        advantages (torch.Tensor): shape (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    for t in range(T):
        advantage = 0.0
        for k in range(1, T - t + 1):
            # Bug: using r_{t+1} instead of r_{t+2} in gamma powers
            if t + 1 < T:
                reward_term = (gamma ** 2) * rewards[t + 1]  # should be rewards[t + k]
            else:
                reward_term = 0.0
            delta = reward_term + gamma * values[t + k] - values[t + k - 1]
            # Bug: weighting factor w_k = lambda^k instead of (1-lambda)*lambda^{k-1}
            w_k = lmbda ** k
            advantage += w_k * delta
        advantages[t] = advantage
    return advantages

# Sample inputs
rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
values = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])  # length T+1
gamma = 0.9
lmbda = 0.8

adv = compute_gae_buggy(rewards, values, gamma, lmbda)
print("Buggy advantages:", adv)