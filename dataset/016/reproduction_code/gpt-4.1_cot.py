import os
import json
import tensorflow as tf
import time

def get_tf_config(cluster_spec, task_type, task_index):
    """Helper to generate TF_CONFIG environment variable."""
    tf_config = {
        "cluster": cluster_spec,
        "task": {"type": task_type, "index": task_index},
        "environment": "cloud"
    }
    return json.dumps(tf_config)

def create_synthetic_dataset(batch_size, input_shape=(224, 224, 3), num_classes=1000):
    """Create synthetic dataset for benchmarking."""
    def gen():
        while True:
            images = tf.random.uniform(shape=(batch_size,) + input_shape, dtype=tf.float32)
            labels = tf.random.uniform(shape=(batch_size,), maxval=num_classes, dtype=tf.int32)
            yield images, tf.one_hot(labels, num_classes)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((batch_size,) + input_shape, (batch_size, num_classes))
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def main():
    # Cluster configuration - replace with your actual IPs or hostnames and ports
    cluster = {
        "worker": ["ip1:12345", "ip2:12345"]  # Use your real IPs and ports
    }

    # You must set these environment variables externally or uncomment below lines to simulate:
    # os.environ["TF_CONFIG"] = get_tf_config(cluster, "worker", 0)  # For worker 0
    # os.environ["TF_CONFIG"] = get_tf_config(cluster, "worker", 1)  # For worker 1

    # Parse TF_CONFIG from environment
    tf_config = os.environ.get("TF_CONFIG")
    if not tf_config:
        raise RuntimeError("TF_CONFIG environment variable is not set")

    tf_config_json = json.loads(tf_config)
    task_info = tf_config_json["task"]
    task_type = task_info["type"]
    task_index = task_info["index"]

    print(f"Starting worker {task_type} #{task_index}")

    # Create MultiWorkerMirroredStrategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Hyperparameters
    batch_size_per_replica = 32
    num_classes = 1000
    input_shape = (224, 224, 3)
    epochs = 1
    steps_per_epoch = 100

    # Compute global batch size
    num_replicas = strategy.num_replicas_in_sync
    global_batch_size = batch_size_per_replica * num_replicas

    print(f"Global batch size: {global_batch_size}, Replicas: {num_replicas}")

    # Create dataset
    dataset = create_synthetic_dataset(global_batch_size, input_shape, num_classes)
    dataset = dataset.take(steps_per_epoch)

    with strategy.scope():
        model = tf.keras.applications.ResNet50(weights=None, input_shape=input_shape, classes=num_classes)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    print(f"Starting training on worker {task_index}")

    start_time = time.time()
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
    end_time = time.time()

    print(f"Worker {task_index} finished training. Time taken: {end_time - start_time:.2f} sec")

if __name__ == "__main__":
    main()