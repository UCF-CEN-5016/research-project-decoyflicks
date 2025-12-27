import keras_cv
from tensorflow import keras

# Attempt to create a KerasCV model with incompatible config
try:
    model = keras_cv.models.StableDiffusion(
        img_width=512,
        img_height=512,
        jit_compile=False
    )
except Exception as e:
    print(f"Error creating model: {e}")

# Suggested workaround (if known)
print("\nTry installing specific versions:")
print("!pip install keras-cv==0.5.1 tensorflow==2.11.0")