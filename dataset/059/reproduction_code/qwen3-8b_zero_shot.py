import tensorflow as tf
from keras_cv import models

model = models.ImageClassifier(model='efficientnetb0', include_preprocessing=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(tf.data.Dataset.from_tensor_slices(([[[0.5]*224 for _ in range(224)] for _ in range(3)], [[0]])), epochs=1)