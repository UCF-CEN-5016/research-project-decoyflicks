import tensorflow as tf
import pathlib
import tensorflow.keras.utils as keras_utils

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = keras_utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    print(model_dir)
    model = tf.saved_model.load_v2(str(model_dir))
    model.signatures["serving_default"]
    return model

model = load_model('ssd_mobilenet_v1_coco')