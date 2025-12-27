import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Softmax
from tensorflow.keras.models import Model

# Simplified NER model setup
inputs = Input(shape=(10,))  # Sequence length 10
x = Dense(64, activation='relu')(inputs)
outputs = Dense(4, activation='softmax')(x)  # 4 entity classes

ner_model = Model(inputs=inputs, outputs=outputs)

# Sample data - sequence length 10, 4 classes
train_data = np.random.rand(100, 10)
train_labels = np.random.randint(0, 4, size=(100, 10))

# Compile and fit - triggers the error
ner_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# The error occurs here during graph execution
try:
    ner_model.fit(train_data, train_labels, epochs=2, batch_size=16)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Message: {str(e)}")

# Sample prediction that would also fail
sample_input = np.random.rand(1, 10)
try:
    prediction = ner_model.predict(sample_input)
except Exception as e:
    print(f"Prediction error: {type(e).__name__}")