import tensorflow as tf
import tf2onnx

from tf_keras.src.dtensor import lazy_variable  # This triggers the error

# The error occurs during import, so no further code is needed
print("If you see this, the import succeeded (unexpected)")

model = ...  # Your MobileNetV4 model
tf2onnx.convert.from_keras(model, output_path="model.onnx")