import tensorflow as tf

def build_sequential_model():
    """Builds a simple Sequential Keras model."""
    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        layers.Dense(32),
        layers.Dense(10)
    ])
    return model

def convert_using_signatures(model):
    """
    Attempts conversion using explicit signatures and I/O shapes.
    Mirrors the original intent of constructing a TFLiteConverter with signatures.
    """
    converter = tf.lite.TFLiteConverter(
        model_signatures=model.signatures.get('default') if hasattr(model, 'signatures') else None,
        input_shapes={'input_1': (None, 784)},
        output_shapes={'dense_1': (None, 32), 'dense_2': (None, 10)}
    )
    tflite_model = converter.convert()
    return tflite_model

def convert_from_saved_model(saved_model_path):
    """Creates a converter from a SavedModel directory and converts it."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    return tflite_model

def convert_with_dynamic_input(model, input_layer_name):
    """
    Converts using model.signatures and a dynamic input shape for the specified input layer name.
    Keeps the same assumption that a variable-length sequence dimension may exist.
    """
    converter = tf.lite.TFLiteConverter(
        model_signatures=model.signatures if hasattr(model, 'signatures') else None,
        input_shapes={input_layer_name: (1, -1)}
    )
    tflite_model = converter.convert()
    return tflite_model

def convert_with_include_input_shape(saved_model_path):
    """
    Uses from_saved_model with include_input_shape=True when supported by the TF version.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_path,
        include_input_shape=True
    )
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(tflite_bytes, output_path):
    """Saves TFLite model bytes to a file."""
    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)

def run_tflite_inference_from_buffer(tflite_bytes, sample_input):
    """
    Loads a TFLite model from bytes into an interpreter, sets the input tensor,
    runs inference, and returns a list of output arrays.
    """
    # Use model_content for buffer-based loading
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare and set the input tensor. Use the first input by default.
    if not input_details:
        raise RuntimeError("No input details available in the interpreter.")
    input_index = input_details[0]['index']

    # Note: sample_input should match the expected dtype and shape of the model input.
    interpreter.set_tensor(input_index, sample_input)

    # Run inference
    interpreter.invoke()

    # Collect outputs
    results = []
    for out in output_details:
        results.append(interpreter.get_tensor(out['index']))
    return results

if __name__ == "__main__":
    # Build a Keras model
    model = build_sequential_model()

    # Example conversion using an explicit signature and I/O shapes
    try:
        tflite_model_a = convert_using_signatures(model)
    except Exception:
        tflite_model_a = None

    # Example conversion from a SavedModel directory
    try:
        tflite_model_b = convert_from_saved_model('path_to_my_model')
    except Exception:
        tflite_model_b = None

    # Example conversion assuming a dynamic sequence length for a named input layer
    try:
        # Replace 'input_layer_name' with the actual input layer name when available
        tflite_model_c = convert_with_dynamic_input(model, input_layer_name='input_1')
    except Exception:
        tflite_model_c = None

    # Example conversion with include_input_shape=True
    try:
        tflite_model_d = convert_with_include_input_shape('path_to_my_model')
    except Exception:
        tflite_model_d = None

    # Save one of the converted models if conversion succeeded
    chosen_tflite = tflite_model_d or tflite_model_c or tflite_model_b or tflite_model_a
    if chosen_tflite is not None:
        save_tflite_model(chosen_tflite, 'model.tflite')

        # Prepare a sample input (ensure it matches the model's expected input shape/dtype)
        sample_input = ...  # Ensure shape matches expected by the model

        # Run inference using the chosen TFLite model buffer
        try:
            inference_results = run_tflite_inference_from_buffer(chosen_tflite, sample_input)
        except Exception:
            inference_results = None
    else:
        inference_results = None