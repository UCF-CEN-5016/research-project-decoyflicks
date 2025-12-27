import numpy as np
import tensorflow as tf
from tensorflow import keras

class DummyLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred)

inputs = keras.Input(shape=(9,), dtype=tf.int64)
x = keras.layers.Embedding(input_dim=100, output_dim=4)(inputs)
x = keras.layers.Dense(3)(x)
outputs = keras.layers.Softmax(axis=-1)(x)
model = keras.Model(inputs, outputs)
model.compile(loss=DummyLoss())

model.save("dummy_ner_model")

converter = tf.lite.TFLiteConverter.from_saved_model("dummy_ner_model")
tflite_model = converter.convert()
with open("dummy_ner_model.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="dummy_ner_model.tflite")
interpreter.allocate_tensors()
sample_input = np.arange(9, dtype=np.int64)
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))
interpreter.invoke()
print(interpreter.get_tensor(output_index))