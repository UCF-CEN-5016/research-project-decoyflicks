import os
import tensorflow as tf
import keras_nlp

os.environ['Keras_Backend'] = 'tensorflow'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

preset_model = 'bert_tiny_en_uncased_sst2'
classifier = keras_nlp.models.BertClassifier.from_preset(preset_model)

sample_input = ['I love modular workflows in keras-nlp']
try:
    output = classifier.predict(sample_input)
except Exception as e:
    print(e)  # Expecting: 'LossScaleOptimizerV3' object has no attribute 'name'