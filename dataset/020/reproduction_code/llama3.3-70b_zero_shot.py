import tensorflow as tf

tf.compat.v1.enable_eager_execution()

class Trainer:
    def eval_step(self, passthrough_logs, logs):
        return passthrough_logs | logs

trainer = Trainer()
passthrough_logs = {"log1": 1, "log2": 2}
logs = {"log3": 3, "log4": 4}

try:
    result = trainer.eval_step(passthrough_logs, logs)
except TypeError as e:
    print(e)