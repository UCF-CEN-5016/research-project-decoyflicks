import tensorflow as tf

# Simulate 'passthrough_logs' with some initial values
passthrough_logs = {'train_loss': 0.8, 'eval_loss': 1.2}

# Simulate new logs from evaluation step
logs = {'validation_accuracy': 0.9, 'best_steps': 500}

# Fix the TypeError by replacing | operator with dictionary merge
merged_logs = {**passthrough_logs, **logs}
print("Merged Logs:", merged_logs)