import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Define a Sequential model for MNIST classification
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=64)

# Evaluate the model before adding embeddings
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy before adding embeddings:', test_acc)

# Create a new Sequential model with an Embedding layer followed by Conv2D layers
embedding_model = models.Sequential()
embedding_model.add(layers.Embedding(input_dim=10, output_dim=32, input_length=784))
embedding_model.add(layers.Reshape((28, 28, 32)))
embedding_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
embedding_model.add(layers.MaxPooling2D((2, 2)))
embedding_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
embedding_model.add(layers.MaxPooling2D((2, 2)))
embedding_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
embedding_model.add(layers.Flatten())
embedding_model.add(layers.Dense(64, activation='relu'))
embedding_model.add(layers.Dense(10))

# Compile the new model
embedding_model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

# Train the new model
history = embedding_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=64)

# Evaluate the model with embeddings
test_loss, test_acc = embedding_model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy after adding embeddings:', test_acc)