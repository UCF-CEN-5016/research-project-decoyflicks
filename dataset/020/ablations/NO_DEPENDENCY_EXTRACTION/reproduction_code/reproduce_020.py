import tensorflow as tf
from official.vision import tfm

distribution_strategy = tf.distribute.MirroredStrategy()
task = {'model': 'ssd', 'params': {}}
exp_config = {'train': {'batch_size': 32}, 'eval': {'batch_size': 32}}
model_dir = '/tmp/model_dir'

model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True
)