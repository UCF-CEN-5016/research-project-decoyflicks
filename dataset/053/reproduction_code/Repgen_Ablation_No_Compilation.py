import os
os.environ['Keras_Backend'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_nlp

# Set mixed precision policy (commented out to avoid AttributeError)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Run the code up to and including the Inference with a pretrained classifier section
# classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")