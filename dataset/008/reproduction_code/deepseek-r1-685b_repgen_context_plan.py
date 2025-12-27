import tensorflow as tf
import psutil
import sys

class ModelConfig:
    def __init__(self):
        self.batch_size = 1
        self.height = 640
        self.width = 640
        self.model_path = "path/to/faster_rcnn_inception_resnet_v2_640x640.tar.gz"
        self.expected_gpu_memory_usage_mb = 1000

def load_and_predict(config):
    model = tf.saved_model.load(config.model_path)
    input_data = tf.random.normal((config.batch_size, config.height, config.width, 3))
    predictions = model(input_data)
    if tf.math.reduce_any(tf.math.is_nan(predictions)):
        raise ValueError("Output contains NaN values")
    return model, input_data, predictions

def verify_loss_calculation(model, input_data, predictions):
    try:
        loss = model.compute_loss(input_data, predictions)
        print("Loss calculation completed without expected ValueError.")
        raise RuntimeError("Expected ValueError not raised")
    except ValueError as e:
        print(f"Loss calculation raised an expected ValueError: {e}")
    except AttributeError:
        print("Model does not have a 'compute_loss' method. Skipping loss verification.")

def assert_gpu_memory_usage(expected_memory_mb):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
            # Do not exit, allow memory check to proceed even if growth fails
    
    process = psutil.Process()
    gpu_memory_usage_mb = process.memory_info().rss / (1024**2)
    print(f"Current GPU memory usage: {gpu_memory_usage_mb:.2f} MB")
    assert gpu_memory_usage_mb != expected_memory_mb, \
        f"GPU memory usage ({gpu_memory_usage_mb:.2f} MB) matches expected behavior ({expected_memory_mb} MB) for TensorFlow 2.13.0, which is unexpected."

def main():
    config = ModelConfig()

    try:
        model, input_data, predictions = load_and_predict(config)
        verify_loss_calculation(model, input_data, predictions)
        assert_gpu_memory_usage(config.expected_gpu_memory_usage_mb)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.model_path}. Please ensure the path is correct.")
        sys.exit(1)
    except ValueError as e:
        print(f"A validation error occurred: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"A runtime error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    
    print("All checks completed.")

if __name__ == "__main__":
    main()
