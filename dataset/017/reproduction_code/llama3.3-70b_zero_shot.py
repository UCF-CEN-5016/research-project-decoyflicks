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
  model = tf.saved_model.load_v2(str(model_dir))
  model.signatures["serving_default"]
    
  return model

def show_inference(detection_model, image_path):
  image_raw = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image_raw, channels=3)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis, ...]

  output_dict = detection_model.signatures["serving_default"](input_tensor)

  print(output_dict)

model_name = 'ssd_mobilenet_v2_2meter_coco_2018_03_29'
model = load_model(model_name)

TEST_IMAGE_PATHS = ['test_image.jpg']
for image_path in TEST_IMAGE_PATHS:
  show_inference(model, image_path)