import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def prepare_dataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

def train_model(model, train_dataset, test_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

# Create dummy data
train_data = tf.random.uniform(shape=(1000, 28, 28))  # 1000 training samples
test_data = tf.random.uniform(shape=(200, 28, 28))    # 200 test samples

# Convert to tf.data.Dataset
train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)

# Define and train the model
model = create_model()
train_model(model, train_dataset, test_dataset, epochs=5)

# Train the model with validation split
model = create_model()
model.fit(train_dataset, epochs=5, validation_split=0.2)

# Train the model with custom validation data
val_dataset = train_dataset.take(100)  # Take first 100 samples as validation
train_dataset = train_dataset.skip(100)
model = create_model()
model.fit(train_dataset, epochs=5, validation_data=val_dataset)