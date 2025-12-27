import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

def create_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3))(inputs)
    model = Model(inputs, x)
    return model

def save_model(model, file_path):
    model.save(file_path)

def load_model_weights(model, file_path):
    model.load_weights(file_path)

model = create_model()
save_model(model, 'model_saved_model')
load_model_weights(model, 'model_saved_model')