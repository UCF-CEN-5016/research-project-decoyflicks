import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer

def train_model():
    model = keras.Sequential([
        layers.InputLayer(input_shape=(128,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

def make_prediction(tflite_model_path, sample_input):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))
    interpreter.invoke()
    prediction_mask = interpreter.get_tensor(output_index)
    return prediction_mask

def tokenize_and_convert_to_ids(input_text):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(input_text, return_tensors='tf', max_length=128, padding='max_length', truncation=True)
    return inputs['input_ids'].numpy()[0]

def main():
    model = train_model()
    convert_model(model)
    sample_input = tokenize_and_convert_to_ids("eu rejects german call to boycott british lamb")
    try:
        make_prediction('model.tflite', sample_input)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()