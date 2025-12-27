import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from object_detection import model_lib_v2
from object_detection import model_main_tf2
from object_detection import exporter_main_v2

class DetectionFromImageModule(tf.Module):
    def __init__(self, model):
        self.model = model

    def __call__(self, input_tensor):
        return self.model(input_tensor)

def main():
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    detection_module = DetectionFromImageModule(model)
    exporter_main_v2.main(['--input_type=image_tensor', '--pipeline_config_path=pipeline.config', '--trained_checkpoint_dir=/path/to/checkpoint', '--output_directory=/path/to/output'])

if __name__ == '__main__':
    main()