import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast(y_true > 0, dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def build_model():
    input_layer = Input(shape=(None,), dtype=tf.int64, name="input_layer")
    embedding = Dense(32)(input_layer)
    output = Dense(5, activation='softmax')(embedding)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss=CustomNonPaddingTokenLoss())
    return model

model = build_model()
model.save("temp_model")

converter = tf.lite.TFLiteConverter.from_saved_model("temp_model")
tflite_model = converter.convert()
with open("temp_model.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="temp_model.tflite")
interpreter.allocate_tensors()

sample_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))
interpreter.invoke()
prediction = interpreter.get_tensor(output_index)