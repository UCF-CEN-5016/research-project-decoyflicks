import torch
from typing import List, Tuple

def compute_shifted_discounted_terms(
    rewards: torch.Tensor, discount: float, lam: float
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    For each time step t, compute:
      term = discount^2 * rewards[t+1] if t+1 exists else 0.0 (as tensor)
      weight = (1 - lam) * (lam ** t) (as tensor)
    Returns a list of (term, weight) tuples.
    """
    n = rewards.size(0)
    discount_sq = discount ** 2
    base_weight = 1.0 - lam

    terms: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for t in range(n):
        if t + 1 < n:
            term = torch.tensor(discount_sq) * rewards[t + 1]
        else:
            term = torch.tensor(0.0)
        weight = torch.tensor(base_weight * (lam ** t))
        terms.append((term, weight))
    return terms

def print_terms(terms: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    print("Correct terms and weights:")
    for term, weight in terms:
        print(f"Term: {term.item()}, Weight: {weight.item()}")

def main() -> None:
    discount = 0.99
    lam = 0.95
    rewards = torch.tensor([1.0, 2.0, 3.0])

    terms = compute_shifted_discounted_terms(rewards, discount, lam)
    print_terms(terms)

if __name__ == "__main__":
    main()