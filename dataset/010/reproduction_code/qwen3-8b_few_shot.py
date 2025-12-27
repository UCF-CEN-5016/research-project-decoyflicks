import tensorflow as tf

class MultiHeadModel(tf.keras.Model):
    def __init__(self):
        super(MultiHeadModel, self).__init__()
        # Segmentation head: predicts class labels
        self.segmentation_head = tf.keras.layers.Dense(1, activation='sigmoid')
        # Detection head: predicts bounding boxes
        self.detection_head = tf.keras.layers.Dense(4)

    def call(self, inputs):
        # Segmentation output is the class prediction
        segmentation_output = self.segmentation_head(inputs)
        # Detection output is the bounding box prediction
        detection_output = self.detection_head(inputs)
        return segmentation_output, detection_output

# Dummy data
train_data = tf.data.Dataset.from_tensor_slices((tf.random.normal([100, 10]), tf.random.uniform([100], 0, 2)))
val_data = tf.data.Dataset.from_tensor_slices((tf.random.normal([20, 10]), tf.random.uniform([20], 0, 2)))

# Compile the model with two loss functions
model = MultiHeadModel()
model.compile(
    optimizer='adam',
    loss={
        'segmentation_output': 'binary_crossentropy',
        'detection_output': 'mse'  # MSE is used for bounding box regression
    }
)

# During training, both outputs are used
model.fit(train_data, epochs=5)

# During evaluation, the model is run with the same data
# But the loss is computed on the wrong output (detection_head)
# This leads to zero loss (since no labels are provided for detection_head)

# To simulate this, we can manually compute the loss on the wrong head
# and ignore the correct one
loss = model.evaluate(val_data, verbose=0)
print("Validation Loss:", loss)