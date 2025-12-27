import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_ner_model():
    inputs = Input(shape=(10,))
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(4, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_and_fit(model, data, labels, epochs=2, batch_size=16):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(data, labels, epochs=epochs, batch_size=batch_size)

def main():
    ner_model = create_ner_model()

    train_data = np.random.rand(100, 10)
    train_labels = np.random.randint(0, 4, size=(100, 10))

    try:
        compile_and_fit(ner_model, train_data, train_labels)
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}")
        print(f"Message: {str(e)}")

    sample_input = np.random.rand(1, 10)
    try:
        prediction = ner_model.predict(sample_input)
    except Exception as e:
        print(f"Prediction error: {type(e).__name__}")

if __name__ == "__main__":
    main()