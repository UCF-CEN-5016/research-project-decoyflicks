import tensorflow as tf
from official import core as tfm

# Set Python version to 3.8 in your environment to match the reported issue.
# Pull the Docker image for TensorFlow Model Garden: 'docker pull tensorflow/tensorflow:2.12.0-gpu-jupyter'
# Run the Docker container with the command: 'docker run --gpus all -it --rm tensorflow/tensorflow:2.12.0-gpu-jupyter'
# Install the required version of TensorFlow Models Official: 'pip install tf-models-official==2.12.0'

distribution_strategy = tf.distribute.MirroredStrategy()
task = {'model': 'your_model_name', 'params': {}}
exp_config = {'train': {'batch_size': 32, 'steps': 1000}, 'eval': {'batch_size': 32, 'steps': 500}}
model_dir = '/tmp/model_dir'

try:
    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True
    )
except Exception as e:
    print(e)