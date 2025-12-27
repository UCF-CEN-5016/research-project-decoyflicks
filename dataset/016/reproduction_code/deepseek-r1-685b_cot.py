import os
import json
import tensorflow as tf
from tensorflow.python.eager import context

def setup_environment():
    """Configure environment variables for multi-worker training."""
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['ip1:6655', 'ip2:6655']  # Replace with actual IPs
        },
        'task': {'type': 'worker', 'index': 0}  # Will be 1 on second node
    })
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust based on your network interface

def create_model():
    """Create a simple ResNet50 model for testing."""
    return tf.keras.applications.ResNet50(weights=None)

def build_dataset(batch_size=256):
    """Create synthetic dataset to eliminate I/O bottlenecks."""
    inputs = tf.random.normal([batch_size, 224, 224, 3])
    labels = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((inputs, labels)).repeat().batch(batch_size)

def train_step(iterator, model, optimizer):
    """Single training step."""
    def step_fn(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss = tf.nn.compute_average_loss(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    return step_fn

def benchmark_scaling():
    """Test scaling performance across nodes."""
    setup_environment()
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    
    with strategy.scope():
        model = create_model()
        optimizer = tf.keras.optimizers.SGD(0.1)
        dataset = build_dataset()
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(dist_dataset)
        
        # Warmup
        for _ in range(10):
            strategy.run(train_step, args=(iterator, model, optimizer))
        
        # Benchmark
        import time
        start = time.time()
        for _ in range(100):
            strategy.run(train_step, args=(iterator, model, optimizer))
        duration = time.time() - start
        print(f'Throughput: {100 * 256 / duration:.1f} examples/sec')

if __name__ == '__main__':
    # Important: This must be called at program start
    context._reset_context()  
    benchmark_scaling()