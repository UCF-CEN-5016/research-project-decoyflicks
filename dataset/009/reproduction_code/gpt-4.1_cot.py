import tensorflow as tf
from object_detection.models import model_builder
from object_detection.utils import config_util

def main():
    # Path to pipeline config (use a sample or your pipeline config path)
    pipeline_config_path = 'path/to/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config'

    # Load pipeline config and build model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']

    # Build detection model (training=False for inference/export mode)
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Try to inspect or access 'outputs' attribute
    try:
        print("detection_model.outputs:", detection_model.outputs)
    except AttributeError as e:
        print("Caught AttributeError:", e)

if __name__ == '__main__':
    main()