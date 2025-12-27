# Example code with corrected imports to run without error after upgrading keras

from keras_cv.models import load_model
from tensorflow.keras import layers, models, utils as keras_utils

model = load_model("path/to/trained/weights.ckpt")

noise_stem = model.get_layer(name="sample stems")
batch_size = 1
height = 256
width = 256
channels = 3

inputs = keras_utils.BatchInput(shape=(batch_size, height, width, channels))

noise = noise_stem(inputs)
noised_images = layers.Add()([inputs, noise])

autoencoder = models.Model(inputs=inputs, outputs=noised_images)

autoencoder.compile(
    loss="mean_squared_error",
    optimizer=lambda: tf.optimizers.Adam(learning_rate=2e-5),
)

autoencoder.fit(
    x=np.concatenate([np.random.randn(batch_size, height, width, channels), inputs]),
    y=inputs,
    epochs=100,
    batch_size=batch_size,
)