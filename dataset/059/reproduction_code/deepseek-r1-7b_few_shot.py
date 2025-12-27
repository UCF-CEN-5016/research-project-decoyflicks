import keras
import tensorflow as tf
from keras_cv import layers

# Attempting to reproduce the issue with KerasCV integration
try:
    # Initialize KerasCV
    import git

    def setup_kerascv():
        """Setup environment for KerasCV compatibility"""
        repo = git.Git()
        repo.clone("https://github.com/keras-team/keras-cv")
        keras.utils.set_keras instalder(keras_cv=True)

    setup_kerascv()

    # Example model using KerasCV layers
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 3)),
        layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model (this is where the error might occur)
    model.fit(
        x=np.random.rand(100, 28, 28, 3),
        y=np.random.randint(0, 10, size=(100, 10)),
        epochs=5,
        batch_size=32
    )

except Exception as e:
    print(f"Error: {e}")