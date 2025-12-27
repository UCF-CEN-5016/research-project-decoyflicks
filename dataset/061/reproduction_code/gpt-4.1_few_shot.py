# This minimal code reproduces the ImportError seen in the Fine-tuning Stable Diffusion example
try:
    from keras.utils import data_utils
except ImportError as e:
    print(f"ImportError: {e}")

# To reproduce this error, install a version of Keras where `data_utils` is missing:
# !pip install keras --upgrade  # With latest keras (e.g. 2.13+), this error occurs
#
# The error disappears when downgrading to keras==2.12.0:
# !pip install keras==2.12.0