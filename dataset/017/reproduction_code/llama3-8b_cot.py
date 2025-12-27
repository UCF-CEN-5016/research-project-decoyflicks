import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img
from google.colab import drive
import os
import pathlib

drive.mount('/content/drive')
!git clone https://github.com/tensorflow/models.git
%cd models/research/object_detection
!mkdir -p /content/output

import tensorflow as tf

# Load the model
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/research/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.saved_model.load_v2(str(model_dir))
  model.signatures["serving_default"]
    
  return model

# Load the detection model
detection_model = load_model('ssdlite_mobilenet_v2_coco')

for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img
from google.colab import drive
import os
import pathlib

drive.mount('/content/drive')
!git clone https://github.com/tensorflow/models.git
%cd models/research/object_detection
!mkdir -p /content/output

import tensorflow as tf

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/research/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  print(model_dir)
  model = tf.saved_model.load_v2(str(model_dir))
  model.signatures["serving_default"]
    
  return model

detection_model = load_model('ssdlite_mobilenet_v2_coco')

for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)