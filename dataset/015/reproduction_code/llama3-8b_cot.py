import tensorflow as tf

# Minimal setup
batch_size = 32
num_classes = 10

def show_batch(batch):
    # Argument `num_of_examples` is never used
    num_of_examples = None
    for example in batch:
        print(example)

# Triggering condition: Create a sample batch
batch = [tf.random.normal((1, 28, 28, 3)) for _ in range(batch_size)]

# Run the buggy function
show_batch(batch)