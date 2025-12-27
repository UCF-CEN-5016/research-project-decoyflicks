import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model():
    base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5", input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_tensorboard_callback():
    return TensorBoard(log_dir='./logs')

def load_data(directory):
    datagen = ImageDataGenerator(rescale=1./255)
    data = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'
    )
    return data

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()

    train_data = load_data('/content/drive/MyDrive/Dog Vision/train/')
    val_data = load_data('/content/drive/Dog Vision/val/')

    model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[tensorboard]
    )
    return model

# Run the training
model = train_model()