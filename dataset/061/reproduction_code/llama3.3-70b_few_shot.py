import sys

# Recent Keras version
try:
    import keras
    print(f"Keras version: {keras.__version__}")
except ImportError as e:
    print(f"Import error: {e}")

# Try to import 'data_utils' from 'keras.utils'
try:
    from keras.utils import data_utils
except ImportError as e:
    print(f"Import error: {e}")

# Attempt to run the Fine-tuning Stable Diffusion example
try:
    # Simulate the example code
    from keras_cv.models import StableDiffusion
    model = StableDiffusion(
        img_height=512, 
        img_width=512, 
        num_train_steps=1000, 
        train_batch_size=16
    )
    print("Model created successfully")
except Exception as e:
    print(f"Error running example: {e}")