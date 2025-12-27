import tensorflow as tf
from official.core import train_lib
from official.vision import imagenet

def main():
    # Force mixed precision policy to float16 to reproduce potential NaN issues.
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except Exception:
        # Older TF versions might use a different API; ignore if unavailable.
        pass

    model_dir = './results_resnet-rs50_tf2'
    config_file = 'configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml'

    params_override = {
        'runtime': {
            'num_gpus': 1,
            'mixed_precision_dtype': 'float16',
            'distribution_strategy': 'one_device',
            'enable_xla': False,
            'loss_scale': None,
        },
        'task': {
            'train_data': {
                'input_path': '/ppusw/datasets/vision/imagenet/tfrecords/train*',
                'global_batch_size': 2,
                'dtype': 'float16'
            },
            'validation_data': {
                'input_path': '/ppusw/datasets/vision/imagenet/tfrecords/valid*',
                'global_batch_size': 2,
                'dtype': 'float16'
            }
        },
        'trainer': {
            'train_steps': 100,
            'validation_steps': 25000,
            # Intentionally set a large initial LR with no warmup to provoke instability
            'optimizer_config': {
                'optimizer': {
                    'type': 'sgd',
                    'sgd': {
                        'momentum': 0.9,
                        'nesterov': False,
                        'decay': 0.0
                    }
                },
                'learning_rate': {
                    'type': 'cosine',
                    'cosine': {
                        'initial_learning_rate': 1.6,
                        'decay_steps': 100,
                        'offset': 0,
                        'alpha': 0.0,
                        'name': 'CosineDecay'
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_learning_rate': 0,
                        'warmup_steps': 0,
                        'name': 'linear'
                    }
                },
                'ema': None
            },
            # keep other trainer settings similar to the original report
            'loss_upper_bound': 1000000.0,
            'steps_per_loop': 640583,
            'summary_interval': 640583,
            'validation_interval': 640583,
            'checkpoint_interval': 640583
        }
    }

    train_lib.run_experiment(
        config_file, params_override, model_dir=model_dir)

if __name__ == '__main__':
    main()