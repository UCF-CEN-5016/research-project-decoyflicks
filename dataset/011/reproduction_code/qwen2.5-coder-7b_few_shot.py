import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Mixed precision policy
mp_policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(mp_policy)

# Configuration constants
INPUT_SHAPE = (160, 160, 3)
NUM_CLASSES = 1001
INITIAL_LEARNING_RATE = 1.6
DECAY_STEPS = 100
FINAL_LEARNING_RATE = 0.0
STATIC_LOSS_SCALE = 128  # reserved for float16 training (not directly used below)

def build_resnet_model(input_shape: tuple = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> Model:
    base_model = ResNet50(weights=None, input_shape=input_shape)
    x = base_model.output
    x = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return Model(inputs=base_model.input, outputs=x)

def cosine_decay_schedule(step):
    return INITIAL_LEARNING_RATE * (1 + np.cos(np.pi * step / DECAY_STEPS)) / 2

def make_optimizer(learning_rate_schedule) -> SGD:
    return SGD(learning_rate=learning_rate_schedule, momentum=0.9, nesterov=True)

def create_synthetic_dataset(num_samples: int = 100, batch_size: int = 2):
    images = np.random.rand(num_samples, *INPUT_SHAPE).astype(np.float16)
    labels = np.random.randint(0, NUM_CLASSES, size=(num_samples,))
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Build model
model = build_resnet_model()

# Optimizer with cosine learning rate schedule
optimizer = make_optimizer(cosine_decay_schedule)

# Compile model
model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(from_logits=False),
    metrics=[SparseCategoricalAccuracy()]
)

# Prepare dataset (replace with real dataset in practice)
train_dataset = create_synthetic_dataset(num_samples=100, batch_size=2)

# Train
model.fit(train_dataset, epochs=100, steps_per_epoch=50)