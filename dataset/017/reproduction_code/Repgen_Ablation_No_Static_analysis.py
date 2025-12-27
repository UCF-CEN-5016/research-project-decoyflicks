import tensorflow as tf
from absl import logging

# Define constants
MODEL_NAME = 'efficientdet-d0'
BASE_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20201221/'
TEST_IMAGE_PATHS = ['path/to/test/image1.jpg', 'path/to/test/image2.jpg']

def load_model(model_name, base_url):
    logging.info(f'Loading model {model_name} from {base_url}')
    model_path = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_name + '.tar.gz',
        untar=True
    )
    saved_model_dir = os.path.join(model_path, 'saved_model')
    logging.info(f'Loading saved model from {saved_model_dir}')
    return tf.saved_model.load_v2(saved_model_dir)

def show_inference(model, image_paths):
    for image_path in image_paths:
        logging.info(f'Processing image: {image_path}')
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, (512, 512))
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        
        logging.info('Running inference')
        predictions = model.signatures['serving_default'](tf.cast(image, tf.float32))
        print(predictions)

# Load the model
model = load_model(MODEL_NAME, BASE_URL)

# Show inference on test images
show_inference(model, TEST_IMAGE_PATHS)