import tensorflow as tf
from object_detection.model_main_tf2 import main

if __name__ == '__main__':
    flags = ['--pipeline_config_path=./ssd_efficientdet_d0_512x512_coco17_tpu-8.config', '--model_dir=training', '--alsologtostderr']
    tf.compat.v1.app.run(main, argv=flags)