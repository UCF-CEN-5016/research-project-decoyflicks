import tensorflow as tf
from keras_cv import models

def build_model():
    model = models.ResNet50V2(input_shape=(224, 224, 3), include_top=True, weights=None)
    return model

def train_model(model, x_train, y_train):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, epochs=1)

model = build_model()
x_train = ...
y_train = ...
train_model(model, x_train, y_train)