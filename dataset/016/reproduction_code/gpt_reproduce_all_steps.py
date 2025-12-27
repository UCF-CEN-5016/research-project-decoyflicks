import os
import sys
import threading
import time
import tensorflow as tf
from official.legacy.image_classification import classifier_trainer
from official.legacy.image_classification import base_configs

def run_worker(task_index, num_gpus, cluster_spec, data_dir, model_dir):
    os.environ["TF_CONFIG"] = str({
        'cluster': {'worker': cluster_spec},
        'task': {'type': 'worker', 'index': task_index}
    }).replace("'", '"')  # TF_CONFIG expects double quotes

    config = base_configs.ExperimentConfig(
        runtime=base_configs.RuntimeConfig(
            distribution_strategy='multi_worker_mirrored',
            worker_hosts=",".join(cluster_spec),
            num_gpus=num_gpus,
            task_index=task_index,
            run_eagerly=True,
            per_gpu_thread_count=1,
            gpu_thread_mode='gpu_private',
            dataset_num_private_threads=1,
        ),
        train_dataset=base_configs.DatasetConfig(dtype='float32'),
        model=base_configs.ModelConfig(model_params={'model_name': ''}),
    )
    config.train_dataset = base_configs.DatasetConfig()
    config.train_dataset.dtype = 'float32'
    config.runtime.num_gpus = num_gpus

    # Setup directories
    os.makedirs(model_dir, exist_ok=True)
    os.environ["DATA_DIR"] = data_dir
    os.environ["MODEL_DIR"] = model_dir

    # Build and compile model inside strategy scope
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = classifier_trainer.get_trivial_model(num_classes=10)
        ds = classifier_trainer.get_trivial_data()
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            run_eagerly=True,
        )
        # Train only a few steps to reproduce
        model.fit(ds, steps_per_epoch=10, epochs=1, verbose=2)

if __name__ == "__main__":
    # Example cluster spec - replace with actual IPs and ports
    cluster_spec = ["ip1:6655", "ip2:6655"]
    num_gpus_per_worker = 2
    data_dir = "/scratch/cdacapp1/sowmya/actual_imagenet/tf_records"
    model_dir = "/home/cdacapp1/scaling/models-master/official/legacy/image_classification/checkpoints"

    threads = []
    for i in range(len(cluster_spec)):
        t = threading.Thread(target=run_worker, args=(i, num_gpus_per_worker, cluster_spec, data_dir, model_dir))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()