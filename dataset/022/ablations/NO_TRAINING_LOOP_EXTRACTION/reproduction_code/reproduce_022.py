import tensorflow as tf
from orbit import actions

def apply_randaugment(input_tensor, level_std):
    # Assuming a RandAugment function exists in the actions module
    return actions.RandAugment(input_tensor, level_std)

def test_randaugment_std():
    level_std = 0
    input_tensor = tf.random.uniform((32, 224, 224, 3))
    
    output = apply_randaugment(input_tensor, level_std)
    std_dev = tf.math.reduce_std(output)
    assert std_dev.numpy() == 1, "Expected standard deviation to be 1, but got {}".format(std_dev.numpy())
    
    level_std = 1
    output = apply_randaugment(input_tensor, level_std)
    std_dev = tf.math.reduce_std(output)
    assert std_dev.numpy() == 1, "Expected standard deviation to be 1, but got {}".format(std_dev.numpy())

if __name__ == '__main__':
    test_randaugment_std()