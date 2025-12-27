import tensorflow as tf
from tensorflow.keras import layers, optimizers

# Define a simple model without BN layers causing issues
def create_model():
    inputs = tf.keras.Input(shape=(None, None, 3))
    x = tf.keras.layers.DepthwiseConv2D(3)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model = create_model()
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')

# Train the model (this step should not trigger the warning if BN layers are excluded)
model.fit(x_train, y_train, epochs=10)