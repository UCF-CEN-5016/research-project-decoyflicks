import os
import numpy as np
from tensorflow.keras import layers

# Path to example dataset (not needed, just dummy)
data_path = "dummy_data"

def build_faulty_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    outputs = layers.Reshape(target_shape=(-1, 1))(inputs)  # Adds an extra dimension
    model = layers.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    input_shape = (256, 256, 3)  # Example input shape
    output_shape = (None, None, 1)  # Faulty label shape
    
    # Generate dummy labels with one extra dimension
    x = np.random.random((1, 256, 256, 3)).astype(np.float32)
    y = np.random.random((1, 256, 256))  # Should NOT have the last dimension
    
    # Apply incorrect reshaping (as per original bug code comments)
    try:
        y = y.reshape((-1, y.shape[-1]))
        y = y.reshape((y.shape[0], -1))
    except ValueError as e:
        print(f"ValueError occurred during label reshaping: {e}")

    # Build model and attempt to fit (this will throw an error)
    model = build_faulty_model(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Try fitting with incorrect labels
    model.fit(x, y, epochs=1)

if __name__ == "__main__":
    main()

if y.shape[-1] != self.num_classes: