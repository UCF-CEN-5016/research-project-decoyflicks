import tensorflow as tf
import numpy as np
import pandas as pd
from official.projects.waste_identification_ml.model_inference.color_and_property_extractor import extract_properties_and_object_masks
import sys
import tarfile
import os

assert tf.__version__ == '2.13.0'
assert sys.version_info[:3] == (3, 10, 12)

model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz'
model_dir = 'faster_rcnn_model'

# Download and extract the model
os.system(f'wget {model_url} -O model.tar.gz')
with tarfile.open('model.tar.gz', 'r:gz') as tar:
    tar.extractall(path=model_dir)

model = tf.saved_model.load(model_dir)

input_image = np.random.rand(640, 640, 3).astype(np.float32)
final_result = {
    'detection_masks_reframed': [np.random.randint(0, 2, (640, 640), dtype=np.bool)],
    'detection_boxes': [[0.1, 0.1, 0.5, 0.5]],
    'detection_classes': [1],
    'detection_scores': [0.9]
}

try:
    list_of_df, cropped_masks = extract_properties_and_object_masks(final_result, height=640, width=640, original_image=input_image)
except ValueError as e:
    print(e)