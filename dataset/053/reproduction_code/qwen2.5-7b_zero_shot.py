import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from keras_nlp.models import BertClassifier

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

BertClassifier.from_preset("bert_tiny_en_uncased_sst2")