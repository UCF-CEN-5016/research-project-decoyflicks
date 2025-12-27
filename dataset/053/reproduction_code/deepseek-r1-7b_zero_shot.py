import tensorflow as tf
from keras_nlp import models, tasks, precomputed, utils

# Minimal Code to Reproduce the Bug Fix

from keras_nlp import tasks, models

def main():
    task = tasks.Task()
    classifier = models.BertClassifier("bert_tiny_en_uncased_sst2")
    
    # Set mixed precision policy after initializing model
    from tensorflow.keras.mixed_precision import configure as mixed_precision_configure
    configure(mixed_float16=True)
    
    # Now the model should initialize correctly without errors
    classifier.predict(["I love modular workflows in keras-nlp"])

if __name__ == "__main__":
    main()