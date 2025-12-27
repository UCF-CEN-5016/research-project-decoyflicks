import tensorflow as tf
from typing import Tuple

def make_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and validation datasets with dummy data."""
    train_features = tf.random.normal([100, 10])
    train_labels = tf.random.uniform([100, 1], 0, 2)
    val_features = tf.random.normal([20, 10])
    val_labels = tf.random.uniform([20, 1], 0, 2)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    return train_dataset, val_dataset

class MultiHeadModel(tf.keras.Model):
    """Multi-head model with a segmentation head and a detection head."""
    def __init__(self):
        super().__init__()
        # Segmentation head: predicts class labels
        self.segmentation_head = tf.keras.layers.Dense(
            1, activation='sigmoid', name='segmentation_output'
        )
        # Detection head: predicts bounding boxes
        self.detection_head = tf.keras.layers.Dense(
            4, name='detection_output'
        )

    def call(self, inputs, training: bool = False):
        seg_out = self.segmentation_head(inputs)
        det_out = self.detection_head(inputs)
        # Return outputs as a tuple to preserve original behavior
        return seg_out, det_out

def build_and_compile_model() -> tf.keras.Model:
    """Instantiate and compile the multi-head model."""
    model = MultiHeadModel()
    model.compile(
        optimizer='adam',
        loss={
            'segmentation_output': 'binary_crossentropy',
            'detection_output': 'mse'
        }
    )
    return model

def main():
    train_ds, val_ds = make_datasets()
    model = build_and_compile_model()

    # Train using both outputs (labels provided only for segmentation head
    # in the datasets, matching the original behavior)
    model.fit(train_ds, epochs=5)

    # Evaluate on the validation dataset (same structure as training data).
    # This mirrors the original code's behavior where the detection head
    # does not receive labels during evaluation.
    eval_loss = model.evaluate(val_ds, verbose=0)
    print("Validation Loss:", eval_loss)

if __name__ == "__main__":
    main()