import tensorflow as tf
from official.vision.research import efficientdet

# Ensure proper TensorFlow version is used (check for updates)
print(tf.__version__)

def create_model EfficientDetD1COCO17:
    model = efficientdet.EfficientNetB0(
        num_classes=90,
        include_top=False,
    )
    
    # Mark all BN layer gamma and beta as trainable
    for block in model.get blocks():
        if isinstance(block, tf.keras.layers.BatchNormalization):
            block.moving_mean = tf.Variable(tf.zeros_like(block.moving_mean), trainable=True)
            block.moving_variance = tf.Variable(tf.ones_like(block.moving_variance), trainable=True)
    
    return model

def main():
    # Prepare your data loaders here
    train_loader = ...  # Implement custom data loading
    val_loader = ...     # Implement custom data loading
    
    with tf.GradientTape() as tape:
        # Forward pass and loss calculation would go here
        loss_value = compute_loss(model, inputs, labels)
    
    # Collect all trainable variables except those excluded below if any
    optimizer = tf.keras.optimizers.Adam()
    
    # Manually specify which variables to include in the gradient computation
    var_list = [v for v in tape.watched_vars if v trainable and 'bn' not in str(v)]
    
    gradients = tape.gradient(loss_value, var_list)
    optimizer.apply_gradients(zip(gradients, var_list))

if __name__ == '__main__':
    main()