import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

image_size = 72
patch_size = 6

# Reproduction steps and analysis are kept the same as they do not require changes.

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
y_train = y_train.astype('int32')
pipeline_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)

def create_vit_model(patch_size=16, num_heads=4, num_layers=2):
    input_layer = layers.Input(shape=(28, 28, 1))
    
    patches = layers.Reshape((28 // patch_size, 28 // patch_size, patch_size * patch_size))(input_layer)
    patch_embeddings = layers.Conv2D(32, kernel_size=patch_size, strides=patch_size)(patches)
    patch_embeddings = layers.Flatten()(patch_embeddings)
    
    positional_encoding = tf.range(start=0, limit=tf.shape(patch_embeddings)[1], dtype=tf.int32)
    pe = positional_encoding[:, tf.newaxis] * tf.constant([1.0 / (10000 ** (2 * i / patch_size)) for i in range(patch_embeddings.shape[-1])])
    pe[:, ::2] = tf.sin(pe[:, ::2])
    pe[:, 1::2] = tf.cos(pe[:, 1::2])
    positional_encoding = layers.Embedding(input_dim=28 // patch_size ** 2, output_dim=patch_embeddings.shape[-1])(positional_encoding)
    
    transformer_encoder = []
    for _ in range(num_layers):
        attn_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)
        norm1 = layers.LayerNormalization(epsilon=1e-6)
        norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        transformer_block = tf.keras.Sequential([
            norm1,
            attn_layer(query=patch_embeddings, value=patch_embeddings, key=patch_embeddings),
            layers.Add(),
            norm2,
            layers.Dense(32 * 4, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32),
            layers.Add()
        ])
        
        transformer_encoder.append(transformer_block)
    
    output_layer = layers.Dense(10, activation='softmax')
    
    model = models.Sequential(transformer_encoder + [output_layer])
    return model

model = create_vit_model(patch_size=4, num_heads=2, num_layers=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

print(model.summary())

history = model.fit(pipeline_train, epochs=1)