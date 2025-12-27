import tensorflow as tf
from official.vision.beta.modeling.backbones import efficientnet

batch_size = 8
height, width = 320, 320

# Create model with problematic variables from warning
model = efficientnet.EfficientNet(model_id='efficientnet-b1')
inputs = tf.keras.layers.Input(shape=(height, width, 3))
features = model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(features)
outputs = tf.keras.layers.Dense(10)(x)
full_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Create data and compile
x_train = tf.random.normal((batch_size, height, width, 3))
y_train = tf.random.uniform((batch_size,), maxval=10, dtype=tf.int32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
full_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# Train to trigger warning
full_model.fit(x_train, y_train, epochs=1)

# Check for target variables
for var in full_model.variables:
    if 'stack_6/block_1' in var.name or 'top_bn' in var.name:
        print(f"Found variable: {var.name}")