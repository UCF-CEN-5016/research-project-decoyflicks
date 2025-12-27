# test_deepspeed_import.py
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

def main(actor_model='facebook/opt-1.3b', reward_model='facebook/opt-350m', deployment_type='single_gpu'):
    print("Running e2e_rlhf with models:", actor_model, reward_model, deployment_type)

if __name__ == "__main__":
    main()