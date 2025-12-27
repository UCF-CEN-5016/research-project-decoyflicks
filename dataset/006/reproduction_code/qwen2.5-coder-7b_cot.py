import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model, Input

def build_simple_model(input_shape=(None, None, 3), num_classes=10):
    """
    Build a simple convolutional model without BatchNormalization layers.
    """
    inputs = Input(shape=input_shape)
    x = layers.DepthwiseConv2D(3)(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes)(x)
    return Model(inputs=inputs, outputs=outputs)

def compile_model(model: tf.keras.Model, learning_rate: float = 0.001):
    """
    Compile the model with Adam optimizer and categorical crossentropy loss.
    """
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy')
    return model

def train_model(model: tf.keras.Model, x_train, y_train, epochs: int = 10, **fit_kwargs):
    """
    Train the model. Passing additional fit kwargs is supported.
    """
    return model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

if __name__ == "__main__":
    model = build_simple_model()
    model = compile_model(model)
    # Train the model (this should not trigger BN-related warnings since BN layers are excluded)
    model.fit(x_train, y_train, epochs=10)