import tensorflow as tf
import numpy as np
import time

# Simulate dataset loading
def simulate_data_loader(data_dir):
    # Simulate data loading with I/O contention
    print(f"Worker {tf.distribute.cluster_resolver.ClusterResolver().task_type} loading data from {data_dir}")
    time.sleep(2)  # Simulate I/O delay
    return np.random.rand(1000, 224, 224, 3)  # Simulated batch of images

# Simulate model training
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_step(model, images):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros((1000, 1)), predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss.numpy()

# Main function to simulate the bug
def main():
    # Simulate multi-worker mirrored strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        num_gpus_per_worker=2,
        worker_hosts=["ip1:6655", "ip2:6655"],
        task_index=0  # Task index for worker 1
    )
    
    with strategy.scope():
        # Simulate data loading with same data_dir for both workers
        data_dir = "/scratch/cdacapp1/sowmya/actual_imagenet/tf_records"
        dataset = simulate_data_loader(data_dir)
        
        model = create_model()
        
        # Simulate training
        start_time = time.time()
        for _ in range(10):
            loss = train_step(model, dataset)
            print(f"Loss: {loss:.4f}")
        total_time = time.time() - start_time
        print(f"Throughput: {10 / total_time:.2f} examples/sec")

if __name__ == "__main__":
    main()