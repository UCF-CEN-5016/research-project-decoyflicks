import numpy as np
import torch

def gae(rewbuf, lenbuf, gamma=0.99, lam=0.97):
    """
    Compute a generalized advantage estimator (GAE)-like array based on the provided
    reward buffer and length buffer. This function preserves the original logic.
    """
    batch_size = lenbuf[0]
    T = lenbuf[1]

    advantage = np.zeros((batch_size, T), dtype=np.float32)

    for i in range(batch_size):
        for j in range(T - 1):
            if j + 2 <= T:
                advantage[i][j] = gamma * rewbuf[i][j + 1] + (gamma ** 2) * rewbuf[i][j + 2]

    if batch_size == 0 or T == 0:
        return advantage

    gae_vals = advantage.copy()
    for k in range(1, T):
        gae_vals[:, k] *= gamma * lam
        if k > 1:
            gae_vals[:, k] += (1.0 - lam) * (lam ** (k))

    return gae_vals.copy()

if __name__ == "__main__":
    batch_size = 4
    sequence_length = 5
    rewbuf = np.random.rand(batch_size, sequence_length)
    lenbuf = [batch_size, sequence_length]

    result = gae(rewbuf, lenbuf)
    print("Corrected GAE Output:", result)