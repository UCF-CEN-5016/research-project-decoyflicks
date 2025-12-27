import tensorflow as tf
import numpy as np

TFLITE_FILENAME = 'model.tflite'


def build_model() -> tf.keras.Model:
    """Create and compile a simple Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1, 1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def export_model_to_tflite(model: tf.keras.Model, filepath: str) -> None:
    """Convert a Keras model to TFLite and write to a file."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filepath, 'wb') as f:
        f.write(tflite_model)


def load_interpreter(filepath: str) -> tf.lite.Interpreter:
    """Load a TFLite model and allocate its tensors."""
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    return interpreter


def format_input_array(array: np.ndarray) -> np.ndarray:
    """Prepare input data for the TFLite interpreter."""
    return np.expand_dims(array, axis=0)


def run_tflite_inference(interpreter: tf.lite.Interpreter, input_array: np.ndarray) -> np.ndarray:
    """Run inference on a TFLite interpreter with provided input array."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def main() -> None:
    model = build_model()
    export_model_to_tflite(model, TFLITE_FILENAME)
    interpreter = load_interpreter(TFLITE_FILENAME)
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    formatted = format_input_array(data)
    output_data = run_tflite_inference(interpreter, formatted)


if __name__ == '__main__':
    main()