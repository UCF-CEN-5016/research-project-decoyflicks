import tensorflow_models as tfm


def get_augment_module():
    """Return the augmentation module from tensorflow_models."""
    return tfm.official.vision.ops.augment


def create_rand_augmentor():
    """Instantiate and return a RandAugment augmentor instance."""
    augment_module = get_augment_module()
    return augment_module.RandAugment()


def main():
    rand_augment = create_rand_augmentor()
    print(rand_augment)


if __name__ == "__main__":
    main()