import numpy as np
from keras_cv import models

# Load pre-trained model
model = models.StableDiffusion()

# Create dummy data
image = np.random.rand(1, 512, 512, 3).astype("float32")
text = ["example text"]

# Fit the model
model.fit(image, text)