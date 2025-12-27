import deepspeed
import torch
from torch import optim
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


class E2E_RLHF:
    def __init__(self, actor_model_name, reward_model_name=None):
        self.actor_model_name = actor_model_name
        self.reward_model_name = reward_model_name

        self.actor = self._load_actor(self.actor_model_name)
        self.reward = self._load_reward(self.reward_model_name) if self.reward_model_name else None

        self._initialize_deepspeed()

    def _load_actor(self, model_name):
        return AutoModelForCausalLM.from_pretrained(model_name)

    def _load_reward(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name)

    def _initialize_deepspeed(self):
        try:
            # Prefer a safe, minimal initialization call for DeepSpeed's distributed utilities.
            deepspeed.init_distributed()
        except Exception:
            # If DeepSpeed is not configured for distributed or initialization fails,
            # keep the failure silent to preserve original behavior.
            pass