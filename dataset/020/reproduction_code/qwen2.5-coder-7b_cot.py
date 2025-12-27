import tensorflow as tf
from typing import Dict, Any

def merge_log_dicts(base_logs: Dict[str, Any], new_logs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a merged dictionary where keys from new_logs override base_logs."""
    return {**base_logs, **new_logs}

def main() -> None:
    # Simulate 'passthrough_logs' with some initial values
    initial_logs: Dict[str, Any] = {'train_loss': 0.8, 'eval_loss': 1.2}

    # Simulate new logs from evaluation step
    evaluation_logs: Dict[str, Any] = {'validation_accuracy': 0.9, 'best_steps': 500}

    merged_logs = merge_log_dicts(initial_logs, evaluation_logs)
    print("Merged Logs:", merged_logs)

if __name__ == "__main__":
    main()