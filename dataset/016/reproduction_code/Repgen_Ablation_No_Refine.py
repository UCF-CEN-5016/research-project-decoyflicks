import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from official.legacy.image_classification.classifier_trainer import main

if __name__ == '__main__':
    flags = [
        '--mode=train_and_eval',
        '--model_type=resnet',
        '--dataset=imagenet',
        f'--model_dir={os.environ["MODEL_DIR"]}',
        f'--data_dir={os.environ["DATA_DIR"]}',
        '--config_file=configs/examples/resnet/imagenet/gpu1.yaml'
    ]
    tf.compat.v1.app.run(main, flags)