import os
import tensorflow as tf
import keras_nlp

os.environ["Keras_Backend"] = 'tensorflow'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

BATCH_SIZE = 32
input_data = tf.random.normal((1000, 128))
labels = tf.random.uniform((1000, 1), maxval=2, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices((input_data, labels)).batch(BATCH_SIZE)

model = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='binary_crossentropy')
try:
    model.fit(dataset, epochs=1)
except Exception as e:
    print(e)