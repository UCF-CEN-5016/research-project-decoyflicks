import tensorflow as tf
import pathlib

# Set up minimal environment
model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
base_url = 'http://download.tensorflow.org/models/research/object_detection/'
model_file = model_name + '.tar.gz'

# Load model
def load_model(model_name):
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.saved_model.load_v2(str(model_dir))
  model.signatures["serving_default"]
    
  return model

# Trigger the bug
detection_model = load_model(model_name)

# Print the model signatures (should be empty)
print(detection_model.signatures)

# Try to use the model (should fail)
image_path = 'path/to/test/image.jpg'
try:
  # Simulate the show_inference function
  print("Trying to use the model...")
  detection_model.signatures["serving_default"](input_image=tf.random.normal([1, 256, 256, 3]))
except Exception as e:
  print("Error:", e)