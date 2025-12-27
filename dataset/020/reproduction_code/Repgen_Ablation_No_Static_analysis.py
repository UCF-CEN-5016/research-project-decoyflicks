# Import necessary libraries
import tensorflow as tf

# Code snippet from `base_trainer.py`
def combine_logs(passthrough_logs, logs):
    print(type(passthrough_logs), type(logs))
    assert isinstance(passthrough_logs, dict) and isinstance(logs, dict), 'Expected dictionaries'
    return passthrough_logs | logs