from transformers import deepspeed

# Import necessary modules with updated paths
from transformers import deepspeed

import torch
from torch import optim

# Update model loading to use compatible modules
class E2E_RLHF:
    def __init__(self, actor_model_name, reward_model_name):
        from transformers import AutoModelForCausalLM
        self.actor = AutoModelForCausalLM.from_pretrained(actor_model_name)
        # Ensure the correct adapter is used with DeepSpeed
        deepspeed.init Brahmapriya