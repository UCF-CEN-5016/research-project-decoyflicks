import tensorflow as tf
from keras_nlp.tokenizers import BertTokenizer
from keras_nlp.samplers import GreedySampler
import numpy as np

def build_tokenizer() -> BertTokenizer:
    return BertTokenizer(
        vocabulary="path/to/vocab.txt",
        pad_token_id=0,
        eos_token_id=2,
        unk_token_id=100
    )

def build_sequential_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 512),
        tf.keras.layers.LSTM(512),
        tf.keras.layers.Dense(10000)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def get_dummy_training_data(batch_size: int = 32, seq_len: int = 10):
    inputs = np.random.randint(0, 10000, (batch_size, seq_len))
    targets = np.random.randint(0, 10000, (batch_size, seq_len))
    return inputs, targets

def train_model(model: tf.keras.Model, inputs: np.ndarray, targets: np.ndarray, epochs: int = 1):
    model.fit(inputs, targets, epochs=epochs)

def generate_from_prompt(model: tf.keras.Model, _input_sentences):
    prompt = tf.constant([[0, 1, 2, 3, 4]])  # Dummy prompt
    sampler = GreedySampler(end_token_id=2)
    generated_tokens = sampler(model, prompt)
    return generated_tokens

def main():
    bert_tokenizer = build_tokenizer()
    seq_model = build_sequential_model()

    train_inputs, train_targets = get_dummy_training_data()
    train_model(seq_model, train_inputs, train_targets, epochs=1)

    # Decoding with invalid parameter (kept as in original)
    generate_from_prompt(seq_model, [[]])

if __name__ == "__main__":
    main()