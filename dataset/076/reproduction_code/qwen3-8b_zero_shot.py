import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

inputs = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3))(inputs)
model = Model(inputs, x)

model.save('model_saved_model')

model.load_weights('model_saved_model')