import tensorflow as tf
from tensorflow.keras.utils import get_file

# Load a model that requires TF1 (e.g., from the TensorFlow models repository)
model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
base_url = 'http://download.tensorflow.org/models/research/object_detection/'
model_file = model_name + '.tar.gz'
model_dir = get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

# Load the model
model = tf.saved_model.load_v2(str(model_dir))
print(model.signatures["serving_default"])

# Attempt to use the loaded model
for image_path in ['path/to/test/image1.jpg', 'path/to/test/image2.jpg']:
    show_inference(model, image_path)