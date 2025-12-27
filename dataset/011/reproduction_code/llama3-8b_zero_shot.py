import tensorflow as tf
from official.vision import imagenet

def main():
    model_dir = './results_resnet-rs50_tf2'
    config_file = 'configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml'
    params_override = {
        'runtime': {
            'num_gpus': 1,
            'mixed_precision_dtype': 'float16',
            'distribution_strategy': 'one_device',
            'enable_xla': False
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
            'validation_steps': 25000
        }
    }

    train_lib.run_experiment(
        config_file, params_override, model_dir=model_dir)

if __name__ == '__main__':
    main()