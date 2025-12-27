import tensorflow as tf
import tensorflow_hub as hub

NUM_EPOCHS = 1

def create_model():
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5", input_shape=(224,224,3))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def create_tensorboard_callback():
    return tf.keras.callbacks.TensorBoard(log_dir='./logs')

train_data = tf.data.Dataset.from_tensor_slices(["drive/MyDrive/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg"]).map(
    lambda x: tf.io.read_file(x)
).batch(1)

val_data = train_data

early_stopping = tf.keras.callbacks.EarlyStopping()

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[tensorboard, early_stopping]
    )
    return model

model = train_model()