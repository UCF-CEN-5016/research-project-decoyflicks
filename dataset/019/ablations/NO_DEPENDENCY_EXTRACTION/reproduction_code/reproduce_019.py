import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_text  # Install tensorflow-text to reproduce the bug

print(tf.version.VERSION)  # Ensure TensorFlow version is 2.x

batch_size = 32
input_dim = 1152  # Assuming 1152 is the sum of feature sizes from the yt8m task
inputs = tf.random.uniform((batch_size, input_dim))
num_classes = 10  # Set the number of classes for the dummy label tensor
labels = tf.random.uniform((batch_size, num_classes), maxval=2, dtype=tf.int32)

# Mock task configuration for YT8MTask
class MockConfig:
    train_data = {
        'feature_sizes': [384, 384, 384],  # Example feature sizes
        'num_classes': num_classes
    }
    model = {}

yt8m_task = tfm.YT8MTask(MockConfig())
model = yt8m_task.build_model()

try:
    import tensorflow_text
except ImportError as e:
    print(e)  # Verify that the ImportError occurs
    assert "undefined symbol: _ZN4absl12lts_2022062320raw_logging_internal21internal_log_functionB5cxx11E" in str(e)

print(tensorflow_text.__version__)  # Check installed version of tensorflow_text