import tensorflow as tf
from tf_keras.src.dtensor import lazy_variable  # This triggers the error

if __name__ == "__main__":
    print("If you see this, the import succeeded (unexpected)")