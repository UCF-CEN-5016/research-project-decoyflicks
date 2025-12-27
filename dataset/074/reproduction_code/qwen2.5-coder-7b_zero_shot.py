import numpy as np

class ActorLossFn:
    def __init__(self):
        pass

    def __call__(self, actor_log_prob, log_probs, advantages, mask):
        return ((log_probs - actor_log_prob) ** 2) * mask

class Test:
    def __init__(self):
        self.actor_loss_fn = ActorLossFn()

    def test(self)
        num_tokens = 10
        start = 5
        end = num_tokens
        log_probs = np.random.rand(num_tokens)
        actor_log_prob = np.random.rand(num_tokens)
        advantages = np.random.rand(num_tokens)
        action_mask = np.random.randint(0, 2, (num_tokens,))
        
        actor_loss = self.actor_loss_fn(actor_log_prob[start:end],
                                        log_probs[start:end],
                                        advantages[start:end],
                                        action_mask[:, start:])

print("Actor loss:", actor_loss)