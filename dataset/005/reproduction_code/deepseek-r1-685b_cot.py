# mobilenet_test_repro.py
import tf_keras  # This triggers the import chain that fails

def test_model_import():
    try:
        from tf_keras.applications import MobileNetV4ConvSmall
        print("Successfully imported MobileNetV4ConvSmall")
    except ImportError as e:
        print(f"Import failed: {e}")

if __name__ == "__main__":
    test_model_import()

# Working version with explicit tensorflow installation
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV4ConvSmall

def test_model_import():
    try:
        model = MobileNetV4ConvSmall(weights=None)
        print("Successfully created MobileNetV4ConvSmall model")
        return model
    except Exception as e:
        print(f"Model creation failed: {e}")

if __name__ == "__main__":
    model = test_model_import()
    if model:
        # Example of saving to ONNX
        import tf2onnx
        model_proto, _ = tf2onnx.convert.from_keras(model)
        with open("mobilenetv4_conv_small.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())
        print("Saved ONNX model")