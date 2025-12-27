# Reproduction Code for TensorFlow 2.x contrib import error
import tensorflow as tf

def trigger_bug():
    try:
        # This is the problematic import that fails in TF 2.x
        from tensorflow.contrib.quantize.python import graph_matcher
        print("Import succeeded (TF 1.x behavior)")
    except ImportError as e:
        print(f"Error reproduced: {e}")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    trigger_bug()