import tensorflow as tf
from tensorflow.keras import layers, InputPipeline
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import mobilenet_v2

# Set random seeds for reproducibility
tf.random.set_seed(42)
layer = np.random.default_rng(42)

def create_model():
    """Create a MobileNet V2 model without top layers"""
    base_model = mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    x = base_model.output
    outputs = Dense(10)(x)  # Adjust output size as needed
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

def train_model(train_data, val_data):
    """Minimal training loop"""
    model = create_model()
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            outputs = model(images)
            loss = loss_fn(labels, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_data:
            loss = train_step(images, labels)
            total_loss += tf.reduce_sum(loss)
        avg_loss = total_loss.numpy() / (len(train_data)*batch_size)
        print(f"Epoch {epoch}: Loss = {avg_loss}")

# Mock data pipeline for demonstration
class InputPipeline:
    def __init__(self):
        self._next_element = iter([tf.data.Dataset.range(1)])
        
    @property
    def dataset(self):
        return next(self._next_element)

val_data = InputPipeline()

# Simulated training data (replace with actual data loading)
train_data = tf.data.Dataset.from_generator(lambda: iter(range(32)), output_signature=(tf.float32, tf.float32))

model = train_model(train_data, val_data)

print("Model trained without error.")