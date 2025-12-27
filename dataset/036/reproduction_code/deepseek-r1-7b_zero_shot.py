def gae(r, gamma, lambda_):
    K = len(r) - 1
    gae = torch.zeros_like(r)
    for k in range(K):
        delta = r[k+1] + gamma * ... # (The rest of the calculation remains unchanged)
        gae[k] = delta
    gae[0] = delta
    
    # Minimal changes:
    min_gae = gamma ** 2
    for k in range(1, K):
        gae[k] = (gae[k-1] * gamma + r[k+1]) * ... # (The rest remains unchanged)
        
    wk = (1 - lambda_) * (lambda_ ** (k))
    
    return gae * wk