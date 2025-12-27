import tensorflow as tf
import os
import json

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': int(os.environ.get('TASK_INDEX', '0'))}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def dataset_fn():
    x = tf.random.uniform([10000, 224, 224, 3])
    y = tf.random.uniform([10000], maxval=1000, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(1000).repeat().batch(GLOBAL_BATCH_SIZE)
    return ds

with strategy.scope():
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224,224,3), classes=1000)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(dataset_fn(), epochs=1, steps_per_epoch=10)