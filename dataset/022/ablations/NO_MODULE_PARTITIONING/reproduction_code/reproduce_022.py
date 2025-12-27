import tensorflow as tf
from official.vision.ops import augment

def test_randaug_standard_deviation():
    batch_size = 2
    height, width = 224, 224
    image = tf.zeros((batch_size, height, width, 3), dtype=tf.uint8)
    bboxes = tf.ones((batch_size, 4), dtype=tf.float32)

    augmenter = augment.RandAugment(num_layers=2, magnitude=10.0, magnitude_std=0.0)
    aug_image, aug_bboxes = augmenter.distort_with_boxes(image, bboxes)

    assert aug_image.shape == (batch_size, height, width, 3)
    assert aug_bboxes.shape == (batch_size, 4)
    std_dev_0 = tf.math.reduce_std(aug_image)

    augmenter = augment.RandAugment(num_layers=2, magnitude=10.0, magnitude_std=1.0)
    aug_image, aug_bboxes = augmenter.distort_with_boxes(image, bboxes)

    assert aug_image.shape == (batch_size, height, width, 3)
    assert aug_bboxes.shape == (batch_size, 4)
    std_dev_1 = tf.math.reduce_std(aug_image)

    print(f"Standard Deviation with magnitude_std=0.0: {std_dev_0.numpy()}")
    print(f"Standard Deviation with magnitude_std=1.0: {std_dev_1.numpy()}")

test_randaug_standard_deviation()