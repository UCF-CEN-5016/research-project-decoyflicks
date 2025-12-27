import multiprocessing
import time
from typing import Tuple

import numpy as np
import tensorflow as tf


def build_model(input_dim: int = 100, hidden_units: int = 128, output_units: int = 10) -> tf.keras.Model:
    """Create and compile a simple Keras model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(hidden_units, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dense(output_units),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_synthetic_dataset(
    num_samples: int = 1000, input_dim: int = 100, num_classes: int = 10, batch_size: int = 32
) -> tf.data.Dataset:
    """Generate a synthetic dataset for demonstration purposes."""
    features = np.random.rand(num_samples, input_dim)
    labels = np.random.randint(0, num_classes, num_samples)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(num_samples).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def run_training_worker(worker_id: int, num_gpus_per_worker: int = 2) -> None:
    """Simulate a training worker that uses MirroredStrategy over multiple GPUs."""
    devices = [f"/device:GPU:{i}" for i in range(num_gpus_per_worker)]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    with strategy.scope():
        model = build_model()

    dataset = create_synthetic_dataset()

    start_time = time.time()
    model.fit(dataset, epochs=1, verbose=1)
    elapsed = time.time() - start_time

    print(f"Worker {worker_id} completed in {elapsed:.2f}s")


def run_workers(num_workers: int = 2) -> None:
    """Launch multiple worker processes to simulate multi-worker training."""
    worker_processes = []
    for wid in range(num_workers):
        proc = multiprocessing.Process(target=run_training_worker, args=(wid,))
        proc.start()
        worker_processes.append(proc)

    for proc in worker_processes:
        proc.join()


if __name__ == "__main__":
    run_workers(num_workers=2)