import tensorflow as tf
from official.projects.waste_identification_ml.model_inference.color_and_property_extractor import extract_properties_and_object_masks
from official.vision.detection.exporter_main_v2 import export_inference_graph

# Set up the environment
tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for testing

# Download pretrained model
model_dir = tf.keras.utils.get_file(
    'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8',
    'http://download.tensorflow.org/models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz',
    untar=True
)

# Prepare sample dataset (dummy data for reproduction)
sample_image = tf.random.uniform((1, 640, 640, 3), minval=0, maxval=255, dtype=tf.int32)

# Create a dummy final_result dictionary
final_result = {
    'detection_masks_reframed': [tf.random.uniform((640, 640), minval=0, maxval=2, dtype=tf.int32)],
    'detection_boxes': [[0.1, 0.1, 0.5, 0.5]],
}

# Extract properties and object masks
height, width = 640, 640
list_of_df, cropped_masks = extract_properties_and_object_masks(final_result, height, width, sample_image.numpy())

# Export the trained model
export_inference_graph(
    input_type='image_tensor',
    pipeline_config_path='path/to/pipeline.config',
    trained_checkpoint_dir='path/to/checkpoints',
    output_directory='path/to/exported_model',
    input_shape=(640, 640, 3)
)