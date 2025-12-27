import numpy as np

def gae_buggy(rewards, values, gamma, lam):
    """
    Buggy GAE calculation:
    - Uses gamma^2 * r_{t+1} instead of r_{t+2}
    - Uses wk = (1 - lam)^k instead of (1-lam)*lam^{k-1}
    """
    T = len(rewards)
    advantages = np.zeros(T)
    for t in range(T):
        adv = 0.0
        for k in range(1, T - t + 1):
            # Bug here: reward index should be t + k, but buggy code uses t + 1 in the gamma powers
            # Let's reproduce the bug exactly:
            # The original bug says line 36 uses gamma^2 * r_{t+1}, which should be gamma^2 * r_{t+2}
            
            # For simplicity, simulate this for k=2 term:
            # We'll implement the entire sum and mimic the buggy indexing for all k >= 2
            if k == 1:
                r_idx = t + 1  # correct: t + 0, but buggy uses t + 1 maybe? We'll keep as is.
            elif k == 2:
                # Here buggy uses r_{t+1} instead of r_{t+2}
                r_idx = t + 1  # should be t + 2
            else:
                r_idx = t + k  # Assume correct for k > 2 for now (to simulate partial bug)
            
            # Cap index to last reward value
            if r_idx >= T:
                r = 0.0
            else:
                r = rewards[r_idx]
            
            # Buggy weight: wk = (1 - lam) ** k instead of (1 - lam) * lam^{k-1}
            wk = (1 - lam) ** k
            
            adv += (gamma ** k) * wk * r
        
        advantages[t] = adv
    
    return advantages

def gae_correct(rewards, values, gamma, lam):
    """
    Correct GAE calculation:
    - Uses correct reward index: r_{t+k}
    - Uses correct weight: wk = (1-lam) * lam^{k-1}
    """
    T = len(rewards)
    advantages = np.zeros(T)
    for t in range(T):
        adv = 0.0
        for k in range(1, T - t + 1):
            r_idx = t + k
            if r_idx >= T:
                r = 0.0
            else:
                r = rewards[r_idx]
            
            wk = (1 - lam) * (lam ** (k - 1))
            adv += (gamma ** k) * wk * r
        
        advantages[t] = adv
    
    return advantages

def main():
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([0.5, 1.0, 1.5, 2.0])  # not used in this minimal example
    gamma = 0.9
    lam = 0.8
    
    adv_buggy = gae_buggy(rewards, values, gamma, lam)
    adv_correct = gae_correct(rewards, values, gamma, lam)
    
    print("Buggy advantages:", adv_buggy)
    print("Correct advantages:", adv_correct)
    print("Difference:", adv_correct - adv_buggy)

if __name__ == '__main__':
    main()