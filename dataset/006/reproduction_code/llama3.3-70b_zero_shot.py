import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization

def create_model():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def main():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)
    inputs = tf.random.normal([1, 256, 256, 3])
    labels = tf.random.normal([1, 256, 256, 32])
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

if __name__ == "__main__":
    main()