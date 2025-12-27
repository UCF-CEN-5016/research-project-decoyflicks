import tensorflow as tf

WEIGHTS_PATH = 'custom_layer.h5'


def build_dense_model(units: int = 32, activation: str = 'relu') -> tf.keras.Sequential:
    return tf.keras.Sequential([tf.keras.layers.Dense(units, activation=activation)])


def save_model_weights(model: tf.keras.Sequential, path: str) -> None:
    model.save_weights(path)


def try_load_weights(model: tf.keras.Sequential, path: str) -> None:
    try:
        model.load_weights(path)
    except Exception as exc:
        print(f"Error: {exc}")


def copy_weights(target: tf.keras.Sequential, source: tf.keras.Sequential) -> None:
    target.set_weights(source.get_weights())


def main() -> None:
    # Create and save a model's weights
    original_model = build_dense_model()
    save_model_weights(original_model, WEIGHTS_PATH)

    # Attempt to load into an identical model and print any error
    identical_model = build_dense_model()
    try_load_weights(identical_model, WEIGHTS_PATH)

    # Alternatively, load weights into a model and then copy them to another model
    loader_model = build_dense_model()
    loader_model.load_weights(WEIGHTS_PATH)

    new_model = build_dense_model()
    copy_weights(new_model, loader_model)


if __name__ == "__main__":
    main()