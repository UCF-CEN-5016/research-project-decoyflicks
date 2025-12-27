import tensorflow as tf

try:
    from tensorflow.contrib.quantize.python import graph_matcher
except ImportError as e:
    print(f"ImportError: {e}")

try:
    import tf_slim
except ImportError as e:
    print(f"ImportError: {e}")

class Exporter:
    def __init__(self):
        try:
            from tensorflow.contrib.quantize.python import graph_matcher
        except ImportError as e:
            print(f"ImportError: {e}")

def main():
    exporter = Exporter()

if __name__ == "__main__":
    main()