import tensorflow as tf
import numpy as np
import time
import multiprocessing

# Minimal model for demonstration
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Simulate multi-worker training (single-node simulation for reproducibility)
def train_worker(task_index, num_gpus_per_worker=2):
    strategy = tf.distribute.MirroredStrategy(devices=[f'/device:GPU:{i}' for i in range(num_gpus_per_worker)])
    
    with strategy.scope():
        model = create_model()
    
    # Simulate training loop
    dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(1000, 100), np.random.randint(0, 10, 1000)))
    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    start_time = time.time()
    model.fit(dataset, epochs=1, verbose=1)
    elapsed = time.time() - start_time
    
    print(f"Worker {task_index} completed in {elapsed:.2f}s")

if __name__ == "__main__":
    # Simulate two workers (in practice, these would run on separate machines)
    processes = []
    
    for i in range(2):
        p = multiprocessing.Process(target=train_worker, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()