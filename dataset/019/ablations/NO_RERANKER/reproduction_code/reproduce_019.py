import tensorflow as tf  # TensorFlow 2.x
import tensorflow_models as tfm  # TensorFlow Models
from tensorflow_text import core  # TensorFlow Text

def simulate_bug():
    try:
        import tensorflow_models as tfm
    except ImportError as e:
        print(e)
        assert '_ZN4absl12lts_2022062320raw_logging_internal21internal_log_functionB5cxx11E' in str(e)

simulate_bug()