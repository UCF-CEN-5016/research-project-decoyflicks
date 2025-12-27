import tensorflow as tf
import pathlib

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/research/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.compat.v1.saved_model.load(str(model_dir), tags='serve')
    
  return model

# Load a pre-trained model
detection_model = load_model('ssd_mobilenet_v2_coco_2018_03_29')

# This will cause empty tensors and warnings
print(detection_model)

# Try to use the model for inference
input_tensor = tf.random.normal([1, 300, 300, 3])
try:
  outputs = detection_model.signatures['serving_default'](input_tensor)
  print(outputs)
except Exception as e:
  print(f"Error: {e}")