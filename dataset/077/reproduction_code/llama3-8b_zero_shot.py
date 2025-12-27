import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

NUM_EPOCHS = 100

def create_model():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

def create_tensorboard_callback():
    return TensorBoard(log_dir='logs')

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()  
    model.fit(x=None, epochs=NUM_EPOCHS, validation_data=None, validation_freq=1, callbacks=[tensorboard, EarlyStopping()])
    return model

model = train_model()