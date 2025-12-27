import tensorflow as tf

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1001)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.6, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.normal([32, 160, 160, 3]), tf.random.uniform([32], minval=0, maxval=1001, dtype=tf.int32))
).batch(2)

# Train the model
for epoch in range(100):
    for batch in train_dataset:
        images, labels = batch
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(learning_rate=1.6, momentum=0.9)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")