import tensorflow as tf

# Ensure TensorFlow version is 2.12
assert tf.__version__ == '2.12'

# Attempt to import required libraries from tensorflow models
from official.nlp.tasks.masked_lm import BertPretrainerV2, ClassificationHead, ClsHeadConfig, DataConfig, PretrainerConfig, as_dict, build_encoder, get_activation, get_data_loader, register_task_cls

# Print a success message if imports are successful
print("Imports were successful.")