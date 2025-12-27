import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the Retina Net model
inputs = Input(shape=(3, 300, 300), name='image_input')
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
outputs = Dense(10)(x)

model = Model(inputs=inputs, outputs=outputs)

# Define custom label encoding
class CustomLabelEncoder(keras.layers.Layer):
    def __init__(self, num_classes):
        super(CustomLabelEncoder, self).__init__()
        self.num_classes = num_classes

    def call(self, inputs):
        # Trigger the bug by returning an invalid index
        return tf.gather(inputs, 3, 63949)

# Wrap the model with custom label encoding
model_with_label_encoder = keras.Sequential([model, CustomLabelEncoder(num_classes=10)])

# Load training data (e.g., from a dataset)
train_data = ...  # replace with your actual training data

# Trigger the bug by fitting the model with the custom label encoder
try:
    model_with_label_encoder.fit(
        train_data,
        epochs=1,
        validation_data=train_data.take(20),
        callbacks=[EvaluateCOCOMetricsCallback(train_data.take(20))]
    )
except InvalidArgumentError as e:
    print(f"Error: {e}")