import tensorflow as tf
import keras_nlp as knlp

PRESET_NAME = "bert_tiny_en_uncased_sst2"

def load_bert_classifier(preset_name: str):
    """Load a BertClassifier from a preset name."""
    return knlp.models.BertClassifier.from_preset(preset_name)

def main():
    # Load the preset model without mixed precision to avoid the error
    model = load_bert_classifier(PRESET_NAME)
    # Run a sample prediction
    model.predict(["I love modular workflows in keras-nlp"])

    # Enable mixed precision if needed
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Reload the classifier with the (potentially) new global policy
    classifier = load_bert_classifier(PRESET_NAME)
    return model, classifier

if __name__ == "__main__":
    main()