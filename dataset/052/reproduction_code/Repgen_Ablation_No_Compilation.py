import collections
from conlleval import evaluate
from datasets import load_dataset
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

def train_model(train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> dict:
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(train_data, epochs=2, validation_data=test_data)

    print(f"Model accuracy on test data: {model.evaluate(test_data)[1] * 100:.2f}%")

    return history.history

def prepare_dataset(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, (32, 32))
    return image, label

# Load dataset
dataset = load_dataset("conll2003")
train_data = dataset["train"].map(lambda x: prepare_dataset(x["tokens"], x["ner_tags"]), batched=True)
test_data = dataset["validation"].map(lambda x: prepare_dataset(x["tokens"], x["ner_tags"]), batched=True)

# Create label mapping
label_counts = collections.Counter(dataset["train"]["ner_tags"])
label_to_index = {label: idx for idx, (label, _) in enumerate(label_counts.most_common())}

# Train model
history = train_model(train_data.shuffle(1024).batch(32), test_data.batch(32))