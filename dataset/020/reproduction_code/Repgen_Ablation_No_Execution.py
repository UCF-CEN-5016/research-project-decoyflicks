import os
from official.vision.detection.tasks import object_detection
import tensorflow as tf

distribution_strategy = tf.distribute.MirroredStrategy()
task = object_detection.ObjectDetectionTask(
    config_path='/path/to/config.yaml'
)
params = task.get_exp_config()
exp_config = params['trainer']
model_dir = '/path/to/model_dir'

from official.core import train_lib
model, eval_logs = train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=params,
    model_dir=model_dir,
    run_post_eval=True
)