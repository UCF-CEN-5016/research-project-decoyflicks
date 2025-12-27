import tensorflow as tf

try:
    # The error occurs due to mixed precision attributes mismatch in optimizer
    from keras_nlp import models, Task
    Task.set_backend(tf.keras.backend)
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Mixed precision setup skipped for stability")