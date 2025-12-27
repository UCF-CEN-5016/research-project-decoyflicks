import keras
from keras.utils import data_utils
import tensorflow as tf

def main():
    try:
        from keras.utils import data_utils
    except ImportError as e:
        print(e)

if __name__ == "__main__":
    main()