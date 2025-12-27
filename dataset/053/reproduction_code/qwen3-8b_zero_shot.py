import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras_nlp

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")