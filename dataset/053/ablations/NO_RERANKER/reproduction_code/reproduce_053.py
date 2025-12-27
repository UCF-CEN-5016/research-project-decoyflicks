import os
import tensorflow as tf
import keras_nlp

os.environ["Keras_Backend"] = 'tensorflow'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

batch_size = 32
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")

input_data = ['I love modular workflows in keras-nlp']
try:
    output = classifier.predict(input_data)
except Exception as e:
    print(e)

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.keras.mixed_precision.global_policy())