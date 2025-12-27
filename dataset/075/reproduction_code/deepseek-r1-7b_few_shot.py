import deepspeed
from transformers import AutoModelForPreTraining

# Try to load model using parameters that trigger unexpected keyword argument
model = AutoModelForPreTraining.from_pretrained(
    "google/colab-roberta-base",
    config=deepspeed DESEP,  # Replace with appropriate configuration
    args=(None, None),
    kwargs=dict(token=None)  # Unrelated argument causing issue
)

print("Error:", deepspeed DESEP)  # Print the error encountered during initialization