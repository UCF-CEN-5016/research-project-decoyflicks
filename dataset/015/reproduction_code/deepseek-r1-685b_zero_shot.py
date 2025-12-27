import tensorflow as tf

def show_batch(num_of_examples, images, boxes, classes):
    for i in range(3):  # Hardcoded instead of using num_of_examples
        print(f"Image {i}: {images[i].shape}, Boxes: {boxes[i]}, Classes: {classes[i]}")

images = [tf.random.uniform((256, 256, 3)) for _ in range(5)]
boxes = [tf.random.uniform((5, 4)) for _ in range(5)]
classes = [tf.random.uniform((5,), maxval=10, dtype=tf.int32) for _ in range(5)]
show_batch(2, images, boxes, classes)