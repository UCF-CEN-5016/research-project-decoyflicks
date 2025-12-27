import tensorflow as tf
from object_detection import exporter_main_v2

# Simulate a model with missing outputs attribute
class DetectionFromImageModule:
    pass

def export_model(model_dir, pipeline_config_path, trained_checkpoint_prefix, output_dir):
    exporter_main_v2.exporter_main(
        model_dir=model_dir,
        pipeline_config_path=pipeline_config_path,
        trained_checkpoint_prefix=trained_checkpoint_prefix,
        output_dir=output_dir
    )

# Simulate export process that triggers the error
export_model(
    model_dir='.',
    pipeline_config_path='config.config',
    trained_checkpoint_prefix='model.ckpt',
    output_dir='exported_model'
)