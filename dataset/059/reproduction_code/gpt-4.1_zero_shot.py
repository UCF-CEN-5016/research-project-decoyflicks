import tensorflow as tf
import keras_cv

ds = tf.data.Dataset.from_tensor_slices((tf.random.uniform((10, 224, 224, 3)), tf.random.uniform((10,), maxval=10, dtype=tf.int32)))
ds = ds.batch(2)

model = keras_cv.models.ResNet50V2(classes=10)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(ds, epochs=1)