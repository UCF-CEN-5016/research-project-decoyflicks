import tensorflow as tf  # TensorFlow dependency
import tensorflow_models as tfm  # TensorFlow Models dependency
import tensorflow_text as tf_text  # TensorFlow Text dependency

def simulate_import_error():
    import tensorflow_models as tfm

if __name__ == "__main__":
    print(tf.version.VERSION)
    print(tf_text.__version__)
    simulate_import_error()