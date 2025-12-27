import os
import json
import tensorflow as tf
from tensorflow.keras import layers
import time
import argparse

# Configuration (simplified from original bug report)
config = {
    'runtime': {
        'distribution_strategy': 'multi_worker_mirrored',
        'worker_hosts': 'localhost:12345,localhost:12346',  # Replace with actual IPs
        'num_gpus': 1,
        'task_index': 0  # Will be set per worker
    }
}

def create_model():
    """Create a simple model"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])
    return model

def create_dataset():
    """Create synthetic dataset"""
    return tf.data.Dataset.from_tensor_slices(
        (tf.random.normal((1024, 224, 224, 3)), 
         tf.random.uniform((1024,), maxval=10, dtype=tf.int32))
    ).batch(32).repeat()

def setup_workers(task_index=0):
    """Setup multi-worker training with synthetic data"""
    # Configure multi-worker strategy
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': config['runtime']['worker_hosts'].split(',')
        },
        'task': {'type': 'worker', 'index': task_index}
    })
    
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    model = create_model()
    
    # Distributed training
    with strategy.scope():
        model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        # Measure throughput
        start_time = time.time()
        model.fit(create_dataset(), steps_per_epoch=100)
        duration = time.time() - start_time
        print(f"Throughput: {100*32/duration:.1f} samples/sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_index', type=int, default=0)
    args = parser.parse_args()
    
    config['runtime']['task_index'] = args.task_index
    setup_workers(args.task_index)