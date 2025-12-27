import tensorflow as tf
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10)
    ])
    return model

def set_mixed_precision():
    policy = tf.keras.mixed_precision.Policy('float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def configure_optimizer():
    initial_learning_rate = 1.6
    decay_steps = 100
    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, alpha=0.0
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=cosine_decay, momentum=0.9)
    return optimizer

def compile_model(model, optimizer):
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

def generate_dummy_data(num_samples):
    x_train = np.random.rand(num_samples, 784).astype(np.float16)
    y_train = np.random.randint(0, 10, size=(num_samples, 1)).astype(np.int32)
    return x_train, y_train

def train_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

if __name__ == "__main__":
    set_mixed_precision()
    
    model = create_model()
    
    optimizer = configure_optimizer()
    
    compile_model(model, optimizer)
    
    x_train, y_train = generate_dummy_data(1000)
    
    train_model(model, x_train, y_train, epochs=10, batch_size=2)