import torch
from transformers import AutoModel, AutoTokenizer

# Define model and tokenizer
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Run the script with desired dependencies
import subprocess
subprocess.run(["python", "e2e_rlhf.py", "--actor-model", model_name, "--reward-model", "facebook/opt-350m", "--deployment-type", "single_gpu"])

print("Script finished")