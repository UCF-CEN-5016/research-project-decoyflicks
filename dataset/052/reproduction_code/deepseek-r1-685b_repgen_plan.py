import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define a simple NER model
def create_ner_model(input_shape=(10,), num_classes=4):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create NER model
ner_model = create_ner_model()

# Generate sample data
train_data = np.random.rand(100, 10)
train_labels = np.random.randint(0, 4, size=(100, 10))

# Compile and fit the model
ner_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
ner_model.fit(train_data, train_labels, epochs=2, batch_size=16)

# Sample prediction
sample_input = np.random.rand(1, 10)
prediction = ner_model.predict(sample_input)