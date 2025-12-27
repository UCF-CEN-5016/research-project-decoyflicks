import tensorflow as tf

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(16, 3)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Activation('relu')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam')  # no loss provided

x = tf.random.normal((1, 224, 224, 3))
y = tf.random.normal((1, 222, 222, 16))

model.fit(x, y, epochs=1)