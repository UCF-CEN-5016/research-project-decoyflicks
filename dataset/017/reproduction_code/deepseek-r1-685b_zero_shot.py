import tensorflow as tf
import pathlib
import urllib.request
import tarfile
import os

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load_v2(str(model_dir))
    return model.signatures["serving_default"]

def show_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = model(input_tensor)
    print(output_dict)

TEST_IMAGE_PATHS = ['test_image.jpg']
detection_model = load_model('ssd_mobilenet_v1_coco_2017_11_17')
for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path)