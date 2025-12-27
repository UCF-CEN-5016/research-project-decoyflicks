import os
import sys
import threading
import tensorflow as tf
from official.legacy.image_classification import classifier_trainer
from official.legacy.image_classification import base_configs

class TrainingConfig:
    def __init__(self, cluster_spec, num_gpus_per_worker, data_dir, model_dir):
        self.cluster_spec = cluster_spec
        self.num_gpus_per_worker = num_gpus_per_worker
        self.data_dir = data_dir
        self.model_dir = model_dir

def setup_tf_config(task_type, task_index, cluster_spec_list):
    os.environ["TF_CONFIG"] = tf.compat.as_text(tf.config.experimental.set_logical_device_configuration(
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)] * tf.config.list_physical_devices('GPU')
    ))
    tf_config = {
        'cluster': {'worker': cluster_spec_list},
        'task': {'type': task_type, 'index': task_index}
    }
    os.environ['TF_CONFIG'] = str(tf_config).replace("'", '"')

def run_worker_training(task_index, config: TrainingConfig):
    setup_tf_config('worker', task_index, config.cluster_spec)

    runtime_config = base_configs.RuntimeConfig(
        distribution_strategy='multi_worker_mirrored',
        worker_hosts=",".join(config.cluster_spec),
        num_gpus=config.num_gpus_per_worker,
        task_index=task_index,
        run_eagerly=True,
        per_gpu_thread_count=1,
        gpu_thread_mode='gpu_private',
        dataset_num_private_threads=1,
    )

    train_dataset_config = base_configs.DatasetConfig(dtype='float32')
    model_config = base_configs.ModelConfig(model_params={'model_name': ''})

    experiment_config = base_configs.ExperimentConfig(
        runtime=runtime_config,
        train_dataset=train_dataset_config,
        model=model_config,
    )

    os.makedirs(config.model_dir, exist_ok=True)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = classifier_trainer.get_trivial_model(num_classes=10)
        dataset = classifier_trainer.get_trivial_data()
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            run_eagerly=True,
        )
        model.fit(dataset, steps_per_epoch=10, epochs=1, verbose=2)

if __name__ == "__main__":
    cluster_spec_example = ["localhost:6655", "localhost:6656"]
    num_gpus_per_worker_example = 0 # Set to 0 for CPU-only, or actual GPU count
    data_directory = "/tmp/data" # Placeholder, adjust as needed
    model_directory = "/tmp/checkpoints" # Placeholder, adjust as needed

    training_configuration = TrainingConfig(
        cluster_spec=cluster_spec_example,
        num_gpus_per_worker=num_gpus_per_worker_example,
        data_dir=data_directory,
        model_dir=model_directory
    )

    worker_threads = []
    for i in range(len(training_configuration.cluster_spec)):
        thread = threading.Thread(target=run_worker_training, args=(i, training_configuration))
        worker_threads.append(thread)
        thread.start()

    for thread in worker_threads:
        thread.join()

    print("Distributed training simulation complete.")
