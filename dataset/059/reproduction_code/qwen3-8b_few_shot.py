import tensorflow as tf  
from keras_cv import models  

# Attempt to use KerasCV model with incompatible dependencies  
model = models.ResNet50V2(input_shape=(224, 224, 3), include_top=True, weights=None)  

# This will fail due to version incompatibility  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  
model.fit(x_train, y_train, epochs=1)