import os
import json
import time
import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    """Create a simple model"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])
    return model

def create_dataset():
    """Create a synthetic dataset"""
    return tf.data.Dataset.from_tensor_slices(
        (tf.random.normal((1024, 224, 224, 3)), 
         tf.random.uniform((1024,), maxval=10, dtype=tf.int32))
    ).batch(32).repeat()

def setup_workers(task_index):
    """Set up multi-worker configuration and perform distributed training"""
    # Configuration
    worker_hosts = ['localhost:12345', 'localhost:12346']  # Replace with actual IPs
    num_workers = len(worker_hosts)
    
    # Set TF_CONFIG environment variable
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': worker_hosts
        },
        'task': {'type': 'worker', 'index': task_index}
    })

    # Multi-worker mirrored strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # Create model and compile within strategy scope
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        # Distributed training
        start_time = time.time()
        model.fit(create_dataset(), steps_per_epoch=100)
        duration = time.time() - start_time
        print(f"Throughput: {100*32/duration:.1f} samples/sec")

if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_index', type=int, default=0)
    args = parser.parse_args()
    
    setup_workers(args.task_index)