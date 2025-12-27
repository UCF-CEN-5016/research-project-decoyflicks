import keras_cv
from tensorflow import keras

def create_model():
    try:
        model = keras_cv.models.StableDiffusion(
            img_width=512,
            img_height=512,
            jit_compile=False
        )
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def suggest_workaround():
    print("\nTry installing specific versions:")
    print("!pip install keras-cv==0.5.1 tensorflow==2.11.0")

# Attempt to create the model
model = create_model()

# Suggest a workaround if model creation fails
if model is None:
    suggest_workaround()