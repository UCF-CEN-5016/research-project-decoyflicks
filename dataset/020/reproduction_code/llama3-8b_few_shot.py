import tensorflow as tf
from tfm.core import base_trainer

# Minimal reproduction code to demonstrate the issue
model_dir = '/path/to/model/directory'
exp_config = {'some_key': 'some_value'}
distribution_strategy = tf.distribute.MirroredStrategy()

task = 'some_task'  # Replace with actual task name

try:
    model, eval_logs = base_trainer.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True)
except TypeError as e:
    print(f"Error: {e}")