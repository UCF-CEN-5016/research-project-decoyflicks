import numpy as np
import torch

def gae(rewbuf, lenbuf, gamma=0.99, lam=0.97):
    batch_size = lenbuf[0]
    T = lenbuf[1]
    
    # Correcting line 36 issue:
    advantage = np.zeros((batch_size, T), dtype=np.float32)
    for i in range(batch_size):
        for j in range(T - 1):
            if j + 2 <= T:  # Ensure we don't go out of bounds
                advantage[i][j] = gamma * rewbuf[i][j+1] + (gamma**2) * rewbuf[i][j+2]
    
    # Correcting line 45 issue:
    if batch_size == 0 or T == 0:
        return advantage
    
    gae = advantage.copy()
    for k in range(1, T):
        gae[:,k] *= gamma * lam
        if k > 1:  # Weight wk should be (1 - lambda) * lambda^{k-1}
            gae[:,k] += (1.0 - lam) * (lam ** (k))
    
    return gae.copy()

# Example usage:
batch_size = 4
sequence_length = 5
rewbuf = np.random.rand(batch_size, sequence_length)
lenbuf = [batch_size, sequence_length]

result = gae(rewbuf, lenbuf)
print("Corrected GAE Output:", result)