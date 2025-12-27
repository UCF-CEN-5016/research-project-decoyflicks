import numpy as np
from keras import layers
from keras.models import Model

# Minimal environment setup
img_height, img_width = 224, 224
num_classes = 37
input_shape = (3, img_height, img_width)

# Triggering conditions: Incorrect label format with extra dimension
labels_train = np.load('oxford_pets_image_segmentation_labels_train.npy')
labels_train = labels_train[:-1, :, :]  # Remove last dimension

# Set up the model architecture
base_model = layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)
x = base_model(layers.Input(shape=input_shape))
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model with sparse categorical cross-entropy loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (not actually necessary for reproducing the bug)
model.fit(np.load('oxford_pets_image_segmentation_train.npy'), labels_train, epochs=1)