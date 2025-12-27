import os
import tensorflow as tf
from official.legacy.image_classification import classifier_trainer

# Set environment variables on both nodes
os.environ['PYTHONPATH'] = "/home/cdacapp1/scaling/models-master"
os.environ['DATA_DIR'] = "/shared/data"
os.environ['MODEL_DIR'] = "/home/cdacapp1/scaling/models-master/official/legacy/image_classification/checkpointss"

# Start Python interpreter on node1
config_node1 = classifier_trainer.get_exp_config_from_flags(['--mode=train_and_eval', '--model_type=resnet', '--dataset=imagenet', '--model_dir={}'.format(os.environ['MODEL_DIR']), '--data_dir={}'.format(os.environ['DATA_DIR']), '--config_file=/path/to/gpu.yaml'])
strategy_node1 = tf.distribute.MultiWorkerMirroredStrategy(config_node1.runtime.worker_hosts, config_node1.runtime.num_gpus)
with strategy_node1.scope():
    model_node1 = classifier_trainer.build_model_from_config(config_node1.model, num_classes=config_node1.dataset.num_classes)
    train_and_eval_fn_node1 = classifier_trainer.create_train_and_eval_fn(strategy_node1, model_node1)
train_and_eval_fn_node1(config_node1.train_dataset, config_node1.eval_dataset, epochs=config_node1.runtime.epochs)

# Start Python interpreter on node2
config_node2 = classifier_trainer.get_exp_config_from_flags(['--mode=train_and_eval', '--model_type=resnet', '--dataset=imagenet', '--model_dir={}'.format(os.environ['MODEL_DIR']), '--data_dir={}'.format(os.environ['DATA_DIR']), '--config_file=/path/to/gpu1.yaml'])
strategy_node2 = tf.distribute.MultiWorkerMirroredStrategy(config_node2.runtime.worker_hosts, config_node2.runtime.num_gpus)
with strategy_node2.scope():
    model_node2 = classifier_trainer.build_model_from_config(config_node2.model, num_classes=config_node2.dataset.num_classes)
    train_and_eval_fn_node2 = classifier_trainer.create_train_and_eval_fn(strategy_node2, model_node2)
train_and_eval_fn_node2(config_node2.train_dataset, config_node2.eval_dataset, epochs=config_node2.runtime.epochs)

# Monitor GPU utilization and throughput