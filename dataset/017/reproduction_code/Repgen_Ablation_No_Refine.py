import tensorflow as tf
from official.utils.flags import core as flags_core
from official.vision.detection.configs import model_builder_config_pb2
from official.vision.detection.modeling.heads import classification_head
from official.vision.detection.ops import preprocess_ops

# Constants
MODEL_NAME = 'movinet'
BASE_URL = 'https://storage.googleapis.com/tensorflow-models/official/detection/'
TEST_IMAGE_PATH = '/path/to/test/image.jpg'

# Function to load model
def load_model(model_dir):
    saved_model_path = tf.keras.utils.get_file(
        fname=MODEL_NAME,
        origin=f'{BASE_URL}{MODEL_NAME}.pb',
        untar=False)
    loaded_model = tf.saved_model.load_v2(saved_model_path)
    return loaded_model.signatures['serving_default']

# Function to perform inference
def show_inference(model, image_path):
    img_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    input_tensor = preprocess_ops.preprocess_image(
        image,
        model.config.eval_config.input_size[0],
        model.config.eval_config.input_size[1],
        model.config.eval_config.input_size[2])
    input_dict = {
        'image': input_tensor
    }
    output_dict = model(input_dict)
    return output_dict

# Main execution
if __name__ == '__main__':
    flags = flags_core.Flags()
    flags.define_string('model_dir', './models/movinet', 'Path to the directory containing the pre-trained model')
    
    # Load model in TF1 mode
    model = load_model(flags.model_dir)
    
    # Perform inference
    output_dict = show_inference(model, TEST_IMAGE_PATH)
    
    # Print output for verification
    print(output_dict)