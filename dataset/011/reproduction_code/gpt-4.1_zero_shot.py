import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def resnet_rs_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.0, epsilon=1e-5)(x)
    x = tf.nn.swish(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.0, epsilon=1e-5)(x)
    x = tf.nn.swish(x)
    x = layers.Conv2D(filters * 4, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.0, epsilon=1e-5)(x)
    if stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = layers.Conv2D(filters * 4, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization(momentum=0.0, epsilon=1e-5)(shortcut)
    x = layers.Add()([x, shortcut])
    x = tf.nn.swish(x)
    return x

def build_resnet_rs50(input_shape=(160,160,3), num_classes=1001):
    inputs = layers.Input(shape=input_shape, dtype='float16')
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization(momentum=0.0, epsilon=1e-5)(x)
    x = tf.nn.swish(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    blocks = [3, 4, 6, 3]

    for i in range(4):
        for j in range(blocks[i]):
            stride = 2 if j == 0 and i !=0 else 1
            x = resnet_rs_block(x, filters[i], stride=stride)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(num_classes, dtype='float32')(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    return models.Model(inputs, outputs)

strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")

with strategy.scope():
    model = build_resnet_rs50()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1.6, decay_steps=100)
    optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.1)

batch_size = 2
num_classes = 1001

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        scaled_loss = loss * (1.0)
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def generate_fake_data(batch_size, num_classes):
    images = tf.random.uniform([batch_size, 160, 160, 3], dtype=tf.float16)
    labels = tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32)
    labels = tf.one_hot(labels, num_classes, dtype=tf.float16)
    return images, labels

for step in range(100):
    imgs, lbls = generate_fake_data(batch_size, num_classes)
    loss = train_step(imgs, lbls)
    tf.print("step", step, "loss", loss)
    if tf.math.is_nan(loss):
        raise RuntimeError("The loss value is NaN")