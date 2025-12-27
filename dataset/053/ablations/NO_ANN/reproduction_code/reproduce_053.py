import os
import tensorflow as tf
import keras_nlp

# Set the Keras backend to 'tensorflow'
os.environ['Keras_Backend'] = 'tensorflow'

# Set mixed precision policy to 'mixed_float16'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

batch_size = 32

# Instantiate the classifier, which is expected to reproduce the bug
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")

input_data = ['I love modular workflows in keras-nlp']
try:
    # Attempt to predict with the classifier
    classifier.predict(input_data)
except Exception as e:
    # Print any exceptions that occur, which may include the bug
    print(e)

# Ensure the correct versions of TensorFlow and Keras are being used
assert tf.__version__ == '2.15.0'
assert keras_nlp.__version__ == '2.15.0'  # Fixed the undefined variable issue by using keras_nlp
assert tf.config.list_physical_devices('GPU') != []

# Print memory info for the GPU
print(tf.config.experimental.get_memory_info('GPU:0'))