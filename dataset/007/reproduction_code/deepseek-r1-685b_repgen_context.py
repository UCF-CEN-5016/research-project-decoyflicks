import os
import sys

os.system('git clone https://github.com/tensorflow/models.git models/research/object_detection')
os.system('pip install tensorflow==2.10.1')
os.system('pip install tensorflow-addons==0.20.0')

config_path = os.path.join('object_detection', 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config')
os.system(f'copy {config_path} models/research/object_detection')

config_util_path = os.path.join('object_detection', 'utils', 'config_util.py')
with open(config_util_path, 'r+', encoding='latin-1') as f:
    content = f.read()
    f.seek(0)
    f.write(content)
    f.truncate()

os.system('python3.9 model_main_tf2.py --pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr')