def geometric_weights(trace_decay: float, terms: int) -> list:
    """Return geometric eligibility trace weights for given number of terms."""
    base = 1.0 - trace_decay
    return [base * (trace_decay ** k) for k in range(terms)]


def weighted_sum(weights: list, rewards: list) -> float:
    """Compute dot product of weights and rewards (pairwise multiplication)."""
    return sum(w * r for w, r in zip(weights, rewards))


def discounted_reward(discount: float, reward: float, power: int) -> float:
    """Apply discount^power to a single reward."""
    return (discount ** power) * reward


def main() -> None:
    discount = 0.99
    trace_decay = 0.95
    reward_1, reward_2 = 10, 20

    weights = geometric_weights(trace_decay, terms=2)  # weights for k=0 and k=1
    v1 = weighted_sum(weights, [reward_1, reward_2])

    v2 = discounted_reward(discount, reward_2, power=2)
    v3 = discounted_reward(discount, reward_2, power=2)

    print(v1, v2, v3)


if __name__ == "__main__":
    main()