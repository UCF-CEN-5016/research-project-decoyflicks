# Minimal reproduction code
import os

# Simulate evaluation logs from model's metrics
log1 = {
    "eval_loss": 0.82,
    "precision": 0.76,
}

# Example log containing additional metrics or other tracking data
log2 = {
    "total_loss": 1.5,
    "num_samples": 100,
}

# Attempt to merge logs using | which is invalid for dictionaries
try:
    merged_log = log1 | log2
except TypeError as e:
    print(f"TypeError occurred: {e}")