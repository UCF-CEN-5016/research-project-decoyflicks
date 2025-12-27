from keras.utils import FeatureSpace  # Fails in Keras 2.11+

# This will raise:
# ImportError: cannot import name 'FeatureSpace' from 'keras.utils'

from keras.layers import FeatureSpace