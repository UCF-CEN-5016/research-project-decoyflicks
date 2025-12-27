import os
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from datasets import load_dataset
from collections import Counter
from conlleval import evaluate

# Assuming TokenAndPositionEmbedding and TransformerBlock are defined elsewhere
# from your_module import TokenAndPositionEmbedding, TransformerBlock, lookup_layer, make_tag_lookup_table

batch_size = 32
num_tags = 10
vocab_size = 20000

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=None)
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype="float32")
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

class NERModel(keras.Model):
    def __init__(self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

conll_data = load_dataset("conll2003")

def export_to_file(export_file_path, data):
    with open(export_file_path, "w") as f:
        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )

os.mkdir("data")
export_to_file("./data/conll_train.txt", conll_data["train"])
export_to_file("./data/conll_val.txt", conll_data["validation"])

def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags

def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)

train_data = tf.data.TextLineDataset("./data/conll_train.txt")
val_data = tf.data.TextLineDataset("./data/conll_val.txt")

train_dataset = (
    train_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)
val_dataset = (
    val_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)

ner_model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)

loss = CustomNonPaddingTokenLoss()
tf.config.run_functions_eagerly(True)
ner_model.compile(optimizer="adam", loss=loss)
ner_model.fit(train_dataset, epochs=10)

sample_input = "eu rejects german call to boycott british lamb"
sample_input = lowercase_and_convert_to_ids(sample_input.split())
sample_input = tf.reshape(sample_input, shape=[1, -1])

output = ner_model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]

mapping = make_tag_lookup_table()
prediction = [mapping[i] for i in prediction]

# Check for unused variable and potential typo
level2_dau_2 = None  # This variable is declared but never used
# The following line may cause confusion if level2_dau_2 was intended to be used
level3_dau_2 = some_function(level2_dau_2)  # This line should be checked

print(prediction)