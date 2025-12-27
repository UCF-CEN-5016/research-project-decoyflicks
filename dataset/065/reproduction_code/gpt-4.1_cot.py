# Step 1: Import and check keras version
import keras
print("Keras version:", keras.__version__)

# Step 2: Try importing FeatureSpace
try:
    from keras.utils import FeatureSpace
    print("FeatureSpace imported successfully")
except ImportError as e:
    print("ImportError:", e)