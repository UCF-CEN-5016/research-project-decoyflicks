import sys
from transformers import AutoActorModelForCausalLM, AutoRewardModelForCausalLM
from torch import optim

def main():
    args = {
        "actor_model": "facebook/opt-1.3b",
        "reward_model": "facebook/opt-350m",
        "deployment_type": "single_gpu"
    }
    
    actor_model = AutoActorModelForCausalLM.from_pretrained(args["actor_model"])
    reward_model = AutoRewardModelForCausalLM.from_pretrained(args["reward_model"])
    
    optimizer = optim.AdamW([{"params": actor_model.parameters()}, {"params": reward_model.parameters()}], lr=1e-5)
    optimizer.zero_grad()
    
    return

if __name__ == "__main__":
    main()