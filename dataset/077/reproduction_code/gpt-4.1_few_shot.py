import tensorflow as tf
import tensorflow_hub as hub

# Simulate dataset with non-existent file paths
def get_dataset():
    # Dataset with filenames that do not exist
    filenames = ["nonexistent_path/image1.jpg", "nonexistent_path/image2.jpg"]
    labels = [0, 1]

    def decode_img(filename):
        img = tf.io.read_file(filename)  # This will raise NotFoundError
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img

    path_ds = tf.data.Dataset.from_tensor_slices(filenames)
    image_ds = path_ds.map(decode_img)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset.batch(2)

train_data = get_dataset()
val_data = get_dataset()

NUM_EPOCHS = 2

def create_model():
    # Load a TF Hub model for classification
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5",
                       input_shape=(224, 224, 3))
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_tensorboard_callback():
    return tf.keras.callbacks.TensorBoard(log_dir="./logs")

early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)

def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()
    model.fit(train_data,
              epochs=NUM_EPOCHS,
              validation_data=val_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])
    return model

model = train_model()