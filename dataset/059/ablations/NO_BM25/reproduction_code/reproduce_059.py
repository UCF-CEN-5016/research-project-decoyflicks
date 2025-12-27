import keras_tuner
import tensorflow as tf
import keras
import numpy as np

np.random.seed(42)

x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, (1000, 1))
x_val = np.random.rand(1000, 28, 28, 1)
y_val = np.random.randint(0, 10, (1000, 1))

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(units=hp.Choice("units", [32, 64, 128]), activation="relu")(x)
        outputs = keras.layers.Dense(10)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

        optimizer = keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        epoch_loss_metric = keras.metrics.Mean()

        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            epoch_loss_metric.update_state(loss)

        for callback in callbacks:
            callback.set_model(model)

        best_epoch_loss = float("inf")

        for epoch in range(2):
            print(f"Epoch: {epoch}")

            for images, labels in train_ds:
                run_train_step(images, labels)

            for images, labels in validation_data:
                run_val_step(images, labels)

            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_state()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        return best_epoch_loss

tuner = keras_tuner.RandomSearch(
    objective=keras_tuner.Objective("my_metric", "min"),
    max_trials=2,
    hypermodel=MyHyperModel(),
    directory="results",
    project_name="custom_training",
    overwrite=True,
)

tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val))