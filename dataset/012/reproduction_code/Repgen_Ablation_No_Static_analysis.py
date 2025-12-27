import tensorflow as tf

# Ensure correct import path for optimizer
from tensorflow.python.keras.optimizers.legacy import Optimizer as LegacyOptimizer

# Code from classifier_trainer.py
class ClassifierTrainer:
    def __init__(self):
        self.optimizer = LegacyOptimizer()

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Usage
trainer = ClassifierTrainer()
images = tf.random.uniform((2, 321, 321, 3), minval=0.0, maxval=1.0)
labels = tf.random.uniform((2,), minval=0, maxval=1000, dtype=tf.int64)
trainer.train_step(images, labels)