# Install dependencies
# !pip install tf-models-official==2.12.0

import tensorflow as tf
from official.vision import model_lib

distribution_strategy = tf.distribute.MirroredStrategy()

exp_config = {
    'task': {
        'model': {
            'type': 'ssd',
            'params': {}
        }
    },
    'trainer': {
        'train_steps': 100,
        'validation_steps': 10
    }
}

model_dir = '/tmp/model_dir'

model, eval_logs = model_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=exp_config['task'],
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True
)