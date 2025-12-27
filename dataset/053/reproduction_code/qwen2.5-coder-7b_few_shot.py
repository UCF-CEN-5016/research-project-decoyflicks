import os
import tensorflow as tf
from keras_nlp.models import BertClassifier
from typing import List, Optional


def set_mixed_precision_policy(policy: str = "mixed_float16") -> None:
    """Configure Keras backend environment and set mixed precision policy."""
    os.environ["KERAS_BACKEND"] = "tensorflow"
    tf.keras.mixed_precision.set_global_policy(policy)


def load_model_and_predict(preset: str, texts: List[str]) -> Optional[object]:
    """Load a BertClassifier from a preset and run prediction on provided texts.
    
    Returns the raw prediction result if successful, otherwise None.
    """
    try:
        model = BertClassifier.from_preset(preset)
        result = model.predict(texts)
        return result
    except AttributeError as e:
        print(f"Error: {e}")
        return None


def main() -> None:
    set_mixed_precision_policy()
    preset_name = "bert_tiny_en_uncased_sst2"
    sample_texts = ["I love modular workflows in keras-nlp"]
    load_model_and_predict(preset_name, sample_texts)


if __name__ == "__main__":
    main()