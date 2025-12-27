import tensorflow as tf
from tensorflow.keras import layers
from keras_cv import models, callbacks

def create_model():
    inputs = layers.Input(shape=(None, None, 3))
    # Output shape: (batch_size, 1, 4) — one box per image
    outputs = layers.Lambda(lambda x: tf.random.normal(shape=(1, 4)))(inputs)
    model = tf.keras.Model(inputs, outputs)
    return model

def create_dataset():
    # Ground truth boxes with shape (batch_size, 2, 4)
    ground_truth_boxes = tf.constant([
        [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]],
        [[0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]],
    ])
    dataset = tf.data.Dataset.from_tensor_slices({
        'boxes': ground_truth_boxes,
        'classes': tf.constant([[0, 1], [2, 3]], dtype=tf.int32)
    })
    return dataset

def train_model(model, train_ds, val_ds):
    callback = callbacks.EvaluateCOCOMetricsCallback(val_ds, "model.h5")
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[callback])

if __name__ == '__main__':
    model = create_model()
    train_ds = create_dataset()
    val_ds = create_dataset()
    train_model(model, train_ds, val_ds)