import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer

# Load model and tokenizer
MODEL_NAME = 'bert-base-uncased'
model = TFAutoModelForTokenClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_to_input_ids(text):
    """Tokenize a single string and return input_ids as tf.int32 tensor shape (1, seq_len)."""
    encoded = tokenizer(text, return_tensors='tf')
    return tf.cast(encoded.input_ids, dtype=tf.int32)

def create_dataset(texts, labels, batch_size=1):
    """
    Create a simple tf.data.Dataset for token classification training.
    Each example is tokenized, labels are converted to per-token one-hot vectors
    and sequences are padded to the same length for batching.
    """
    examples_input_ids = []
    examples_labels = []
    num_labels = getattr(model.config, "num_labels", 1)

    for text, label_seq in zip(texts, labels):
        input_ids = tokenize_to_input_ids(text)        # shape (1, seq_len)
        input_ids = tf.squeeze(input_ids, axis=0)      # shape (seq_len,)
        seq_len = int(input_ids.shape[0])

        # Create per-token one-hot labels; default to zeros and set first token's label
        onehot = tf.zeros([seq_len, num_labels], dtype=tf.float32)
        try:
            first_label_index = int(label_seq[0])
        except Exception:
            first_label_index = 0
        if 0 <= first_label_index < num_labels:
            onehot = tf.tensor_scatter_nd_update(onehot, indices=[[0, first_label_index]], updates=[1.0])

        examples_input_ids.append(input_ids)
        examples_labels.append(onehot)

    # Pad all sequences to the maximum length
    max_len = max(int(t.shape[0]) for t in examples_input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_inputs = []
    padded_labels = []
    for inp, lab in zip(examples_input_ids, examples_labels):
        pad_len = max_len - int(inp.shape[0])
        padded_inp = tf.pad(inp, [[0, pad_len]], constant_values=pad_token_id)
        padded_lab = tf.pad(lab, [[0, pad_len], [0, 0]], constant_values=0.0)
        padded_inputs.append(padded_inp)
        padded_labels.append(padded_lab)

    input_batch = tf.stack(padded_inputs)   # shape (batch, seq_len)
    labels_batch = tf.stack(padded_labels)  # shape (batch, seq_len, num_labels)

    dataset = tf.data.Dataset.from_tensor_slices((input_batch, labels_batch)).batch(batch_size)
    return dataset

def predict_labels_for_text(text, id_to_label):
    """
    Run inference on a single string and return a list of predicted label strings
    for the tokens in the input.
    """
    input_ids = tokenize_to_input_ids(text)           # shape (1, seq_len)
    input_ids = tf.cast(input_ids, dtype=tf.int32)
    # model.predict accepts a dict with input_ids for TF models
    logits = model.predict({'input_ids': input_ids})
    # logits shape: (batch, seq_len, num_labels)
    predicted_ids = tf.argmax(logits, axis=-1).numpy()  # shape (batch, seq_len)
    # Map ids to label names for the first (and only) batch item
    predicted_seq = predicted_ids[0].tolist()
    mapped = [id_to_label.get(int(pid), str(pid)) for pid in predicted_seq]
    return mapped

# Prepare a simple dataset (example)
train_texts = ["eu rejects german call to boycott british lamb"]
train_labels = [[0]]  # simplified example labels

train_dataset = create_dataset(train_texts, train_labels, batch_size=1)

# Compile model (ensure optimizer and loss are compatible)
loss_fn = lambda y_true, y_pred: tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
model.compile(optimizer='adam', loss=loss_fn)

# Train model (single epoch example)
model.fit(train_dataset, epochs=1)

# Prepare mapping from class id to label name
if hasattr(model.config, "id2label") and model.config.id2label:
    id_to_label_map = {int(k): v for k, v in model.config.id2label.items()}
else:
    num_labels = getattr(model.config, "num_labels", 1)
    id_to_label_map = {i: f"LABEL_{i}" for i in range(num_labels)}

# Run a sample inference and print results
result_labels = predict_labels_for_text("eu rejects german call to boycott british lamb", id_to_label_map)
print(result_labels)