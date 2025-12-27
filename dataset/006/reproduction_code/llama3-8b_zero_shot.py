import tensorflow as tf
from efficientdet import EfficientDet

model = EfficientDet(backbone_name='d1', num_classes=81)

train_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 224, 224, 3)), tf.zeros((1, 2))))
validation_dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 224, 224, 3)), tf.zeros((1, 2))))

model.compile(optimizer='adam', loss='categorical_crossentropy')

history = model.fit(train_dataset, epochs=5, validation_data=validation_dataset)