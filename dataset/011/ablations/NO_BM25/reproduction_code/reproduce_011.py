import tensorflow as tf
import numpy as np
import pandas as pd
from official.vision.beta import train

# Dependencies: tensorflow==2.6.3, tensorflow-addons==0.14.0

def main():
    # Set environment variables
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # Prepare the training command parameters
    params_override = {
        'runtime.enable_xla': False,
        'runtime.num_gpus': 1,
        'runtime.mixed_precision_dtype': 'float16',
        'runtime.distribution_strategy': 'one_device',
        'task.train_data.input_path': '/ppusw/datasets/vision/imagenet/tfrecords/train*',
        'task.train_data.global_batch_size': 2,
        'task.train_data.dtype': 'float16',
        'task.validation_data.input_path': '/ppusw/datasets/vision/imagenet/tfrecords/valid*',
        'task.validation_data.global_batch_size': 2,
        'task.validation_data.dtype': 'float16',
        'trainer.train_steps': 100,
        'trainer.validation_steps': 25000,
        'trainer.validation_interval': 640583,
        'trainer.steps_per_loop': 640583,
        'trainer.summary_interval': 640583,
        'trainer.checkpoint_interval': 640583,
        'trainer.optimizer_config.ema': '',
        'trainer.optimizer_config.learning_rate.cosine.decay_steps': 100,
        'trainer.optimizer_config.warmup.linear.warmup_steps': 0
    }

    # Start training
    train.train_and_eval(
        experiment='resnet_rs_imagenet',
        mode='train_and_eval',
        model_dir='./results_resnet-rs50_tf2',
        config_file='configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml',
        params_override=params_override
    )

if __name__ == "__main__":
    main()