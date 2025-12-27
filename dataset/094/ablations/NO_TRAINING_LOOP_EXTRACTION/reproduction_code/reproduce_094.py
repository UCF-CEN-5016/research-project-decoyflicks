import torch
from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.optimizer import get_optimizer
from palm_rlhf_pytorch.ppo import ActorCritic
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.utils import top_k

torch.manual_seed(42)

batch_size = 1
seq_len = 1024
feature_dim = 512

x = torch.randn(batch_size, seq_len, feature_dim)
m = torch.randn(batch_size, seq_len) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

model = PaLM(dim=feature_dim, num_tokens=259, depth=6, causal=True)
actor_critic = ActorCritic(palm=model)

logits, mems_out = actor_critic(x, mask=m, return_mems=True, mems=mems)
print(logits.shape)
for mem in mems_out:
    print(mem.shape)

logits_none, mems_none = actor_critic(x, mask=m, return_mems=True, mems=None)
print(logits_none.shape)