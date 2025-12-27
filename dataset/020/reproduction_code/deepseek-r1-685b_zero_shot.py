import tensorflow as tf
from official.core import base_trainer

class TestTrainer(base_trainer.Trainer):
    def eval_step(self, inputs, model, metrics=None):
        passthrough_logs = {'a': 1}
        logs = {'b': 2}
        return passthrough_logs | logs

trainer = TestTrainer(model_dir='test')
inputs = tf.constant(1)
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
trainer.eval_step(inputs, model)