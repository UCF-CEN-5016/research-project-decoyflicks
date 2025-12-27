import torch

def rotate_queries_and_keys(Q, K):
    position = 0
    return Q / position, K / position

Q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
K = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

rotated_Q, rotated_K = rotate_queries_and_keys(Q, K)
print(torch.isnan(rotated_K).any())