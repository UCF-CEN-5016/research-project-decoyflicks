import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import magnitude

BATCH_SHAPE = (1, 3, 224, 224)
AUG_LEVEL = 0.5
AUG_LEVEL_STD = 0.5
NUM_ITERATIONS = 10


def create_input_tensor(shape):
    return tf.random.uniform(shape)


def build_rand_augment(level, level_std):
    return tf.data.AugmentationRandAugment(level=level, level_std=level_std)


def run_augmentation(augmenter, input_tensor, iterations):
    augmented = None
    for _ in range(iterations):
        augmented = augmenter(input_tensor)
    return augmented


def main():
    input_tensor = create_input_tensor(BATCH_SHAPE)
    augmenter = build_rand_augment(AUG_LEVEL, AUG_LEVEL_STD)
    augmented_tensor = run_augmentation(augmenter, input_tensor, NUM_ITERATIONS)
    return augmented_tensor


if __name__ == "__main__":
    main()