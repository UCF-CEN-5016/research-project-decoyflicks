import os
import sys
import tensorflow as tf
from tensorflow import contrib  # THIS LINE SHOULD CAUSE THE ERROR

# Set up minimal environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add triggering conditions
if __name__ == '__main__':
    print("Error reproduction script started.")
    try:
        from tensorflow.contrib.quantize.python import graph_matcher  # THIS LINE SHOULDN'T RUN
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

print("Script completed without errors.")