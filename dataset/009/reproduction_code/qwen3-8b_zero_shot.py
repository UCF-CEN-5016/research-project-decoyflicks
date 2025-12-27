import tensorflow as tf
from object_detection import exporter_main_v2

# Simulate a model with missing outputs attribute
class DetectionFromImageModule:
    pass

# Simulate export process that triggers the error
exporter_main_v2.exporter_main(
    model_dir='.',
    pipeline_config_path='config.config',
    trained_checkpoint_prefix='model.ckpt',
    output_dir='exported_model'
)