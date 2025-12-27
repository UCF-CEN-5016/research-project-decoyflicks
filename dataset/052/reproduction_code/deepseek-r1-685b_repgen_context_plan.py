import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_ner_model(input_shape=(10,), num_classes=4):
    inputs = Input(shape=input_shape)  # Sequence length 10
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)  # 4 entity classes
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Simplified NER model setup
    ner_model = build_ner_model()

    # Sample data - sequence length 10, 4 classes
    train_data = np.random.rand(100, 10)
    train_labels = np.random.randint(0, 4, size=(100, 10))

    # Compile and fit
    ner_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Training
    try:
        ner_model.fit(train_data, train_labels, epochs=2, batch_size=16)
    except Exception as e:
        print(f"Error occurred during training: {type(e).__name__}")

    # Sample prediction
    sample_input = np.random.rand(1, 10)
    try:
        prediction = ner_model.predict(sample_input)
    except Exception as e:
        print(f"Prediction error: {type(e).__name__}")